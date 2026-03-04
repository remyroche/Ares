import re

with open('extreme_price_movements/features.py', 'r') as f:
    content = f.read()

# Make sure concurrent.futures is imported
if 'from concurrent.futures import ProcessPoolExecutor, as_completed' not in content:
    content = content.replace("import pandas as pd", "import pandas as pd\nfrom concurrent.futures import ProcessPoolExecutor, as_completed")

# Add a helper function at the top level for perp features
helper_func = """
def _compute_perp_col(sym, df_sym):
    from extreme_price_movements.perp_features import compute_features as compute_perp_features
    try:
        sym_feats = compute_perp_features(df_sym)
        return sym, sym_feats, None
    except Exception as exc:
        return sym, None, exc
"""

if "_compute_perp_col" not in content:
    content = content.replace("def _compute_features_impl(panel, mkt_gates, cfg):", helper_func + "\ndef _compute_features_impl(panel, mkt_gates, cfg):")

old_code = """            perp_buffers: dict[str, dict[str, pd.Series]] = {}
            for sym in perp_price_panel.columns:
                df_sym = pd.DataFrame(
                    {
                        "funding_rate": funding_aligned[sym],
                        "open_interest": oi_aligned[sym],
                        "perp_price": perp_price_panel[sym],
                        "spot_price": spot_price_panel[sym],
                        "volume": volume_panel[sym],
                        "close": perp_price_panel[sym],
                    },
                    index=perp_price_panel.index,
                )
                try:
                    sym_feats = compute_perp_features(df_sym)
                except Exception as exc:
                    tprint(f"WARN perp feature compute failed for {sym}: {exc}")
                    continue

                for raw_name, ser in sym_feats.items():
                    fname = f"perp_{raw_name}"
                    if fname not in perp_buffers:
                        perp_buffers[fname] = {}
                    perp_buffers[fname][sym] = ser"""

new_code = """            perp_buffers: dict[str, dict[str, pd.Series]] = {}

            import multiprocessing
            max_workers = min(8, multiprocessing.cpu_count())

            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                for sym in perp_price_panel.columns:
                    df_sym = pd.DataFrame(
                        {
                            "funding_rate": funding_aligned[sym],
                            "open_interest": oi_aligned[sym],
                            "perp_price": perp_price_panel[sym],
                            "spot_price": spot_price_panel[sym],
                            "volume": volume_panel[sym],
                            "close": perp_price_panel[sym],
                        },
                        index=perp_price_panel.index,
                    )
                    futures.append(executor.submit(_compute_perp_col, sym, df_sym))

                for future in as_completed(futures):
                    sym, sym_feats, exc = future.result()
                    if exc is not None:
                        tprint(f"WARN perp feature compute failed for {sym}: {exc}")
                        continue

                    if sym_feats is not None:
                        for raw_name, ser in sym_feats.items():
                            fname = f"perp_{raw_name}"
                            if fname not in perp_buffers:
                                perp_buffers[fname] = {}
                            perp_buffers[fname][sym] = ser"""

content = content.replace(old_code, new_code)

with open('extreme_price_movements/features.py', 'w') as f:
    f.write(content)
