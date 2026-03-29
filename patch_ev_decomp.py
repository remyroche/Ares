with open('extreme_price_movements/training.py', 'r') as f:
    content = f.read()

old_logic = '''            _ps_buckets = {}
            for _b in ["long_mr", "long_tf", "short_mr", "short_tf"]:
                _util_key = f"{_b}_utility"'''

new_logic = '''            _ps_buckets = {}
            _strats = get_strategies(cfg)
            for _strat in _strats:
                _b = f"{_strat['trade_side']}_{_strat['strategy_id']}"
                _util_key = f"{_b}_utility"'''

if old_logic in content:
    content = content.replace(old_logic, new_logic)
    with open('extreme_price_movements/training.py', 'w') as f:
        f.write(content)
    print("Patched EV Decomposition bucket iteration!")
else:
    print("Could not find EV Decomposition loop!")
