import re

with open('extreme_price_movements/lgbm_based_mask_generation.py', 'r') as f:
    content = f.read()

# Pass horizon to make_regime_weights in InteractionModel.train_fold
# Need to find the signature of train_fold first:
# def train_fold(
#        self,
#        X_tr,
#        y_tr,
#        symbol_id_tr,
#        X_va,
#        y_va,
#        fold_id: int,
#        seed: int,
#        target_type: str = "quantile",
#    ):
# Then find where make_regime_weights is called inside train_fold

content = re.sub(
    r"def train_fold\(\n        self,\n        X_tr,\n        y_tr,\n        symbol_id_tr,\n        X_va,\n        y_va,\n        fold_id: int,\n        seed: int,\n        target_type: str = \"quantile\",",
    "def train_fold(\n        self,\n        X_tr,\n        y_tr,\n        symbol_id_tr,\n        X_va,\n        y_va,\n        fold_id: int,\n        seed: int,\n        target_type: str = \"quantile\",\n        horizon: int = 10,",
    content
)

content = re.sub(
    r"sample_weight = make_regime_weights\(y_tr, symbol_id_tr\)",
    "sample_weight = make_regime_weights(y_tr, symbol_id_tr, horizon=horizon)",
    content
)

# And now we need to pass horizon from where train_fold is called in run_mining_stage
# Wait, let's search for "model_engine.train_fold("
with open('extreme_price_movements/lgbm_based_mask_generation.py', 'w') as f:
    f.write(content)
