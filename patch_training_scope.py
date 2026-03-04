import re

# Let's fix training.py to sort by timestamp before calling _train_ps_models.

with open('extreme_price_movements/training.py', 'r') as f:
    content = f.read()

# find: _ps_df = pd.concat(_ps_rows, axis=0, ignore_index=True)
search = '''                    _ps_df = pd.concat(_ps_rows, axis=0, ignore_index=True)'''
replace = '''                    _ps_df = pd.concat(_ps_rows, axis=0, ignore_index=True)
                    if "timestamp" in _ps_df.columns:
                        _ps_df = _ps_df.sort_values("timestamp").reset_index(drop=True)'''

content = content.replace(search, replace)

with open('extreme_price_movements/training.py', 'w') as f:
    f.write(content)
