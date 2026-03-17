from extreme_price_movements.config import CFG
from extreme_price_movements.offline_optimisers.params_store import apply_offline_optimizer_best_params
from extreme_price_movements.strategy_registry import get_strategies

cfg = apply_offline_optimizer_best_params(CFG)
strats = get_strategies(cfg)
print(f"Loaded {len(strats)} strategies.")
if strats:
    print(strats[0])
