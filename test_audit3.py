import extreme_price_movements.training_utils as tu
import extreme_price_movements.config as cfg

res = tu.audit_feature_coverage(["ret1h", "ret2h", "my_new_feature"], cfg.CFG)
print(res)
