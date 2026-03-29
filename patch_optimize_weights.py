with open('extreme_price_movements/training.py', 'r') as f:
    content = f.read()

old_logic = """def _optimize_training_sample_weights(
    df: pd.DataFrame,
    X_frame: pd.DataFrame,
    y_ret: np.ndarray,
    label_times: pd.DataFrame,
    base_weights: np.ndarray,
    cfg: dict,
    stage: str,
    extra_components: dict | None = None,
) -> np.ndarray:"""

new_logic = """def _optimize_training_sample_weights(
    df: pd.DataFrame,
    X_frame: pd.DataFrame,
    y_ret: np.ndarray,
    label_times: pd.DataFrame,
    base_weights: np.ndarray,
    cfg: dict,
    stage: str,
    extra_components: dict | None = None,
    strategy: dict | None = None,
) -> np.ndarray:"""

content = content.replace(old_logic, new_logic)

old_logic_2 = """            if "_mr_" in str(stage).lower():
                _tau_mfe = float(cfg.get("mr_weight_mfe_tau", 1.0))
                _tau_mae = float(cfg.get("mr_weight_mae_tau", 1.0))
                _mfe_rel = np.clip(np.maximum(mfe_v, 0.0) / (tp_v + 1e-9), 0.0, 3.0)
                _mae_rel = np.clip(np.maximum(mae_v, 0.0) / (sl_v + 1e-9), 0.0, 3.0)
                _mfe_score = np.clip(_mfe_rel / max(_tau_mfe, 1e-6), 0.0, 1.0)
                _mae_score = 1.0 - np.clip(_mae_rel / max(_tau_mae, 1e-6), 0.0, 1.0)
                _mr_path_w = np.clip(0.5 + 0.5 * (_mfe_score + _mae_score), 0.25, 1.50)
                w_trade_quality = (
                    np.asarray(w_trade_quality, dtype=np.float64) * _mr_path_w
                )
                w_trade_quality = w_trade_quality / max(
                    float(np.mean(w_trade_quality)), 1e-12
                )"""

new_logic_2 = """            is_mr_strat = strategy.get("is_mr", False) if strategy else "_mr_" in str(stage).lower()
            if is_mr_strat:
                _tau_mfe = float(cfg.get("mr_weight_mfe_tau", 1.0))
                _tau_mae = float(cfg.get("mr_weight_mae_tau", 1.0))
                _mfe_rel = np.clip(np.maximum(mfe_v, 0.0) / (tp_v + 1e-9), 0.0, 3.0)
                _mae_rel = np.clip(np.maximum(mae_v, 0.0) / (sl_v + 1e-9), 0.0, 3.0)
                _mfe_score = np.clip(_mfe_rel / max(_tau_mfe, 1e-6), 0.0, 1.0)
                _mae_score = 1.0 - np.clip(_mae_rel / max(_tau_mae, 1e-6), 0.0, 1.0)
                _mr_path_w = np.clip(0.5 + 0.5 * (_mfe_score + _mae_score), 0.25, 1.50)
                w_trade_quality = (
                    np.asarray(w_trade_quality, dtype=np.float64) * _mr_path_w
                )
                w_trade_quality = w_trade_quality / max(
                    float(np.mean(w_trade_quality)), 1e-12
                )"""

if old_logic_2 in content:
    content = content.replace(old_logic_2, new_logic_2)
    with open('extreme_price_movements/training.py', 'w') as f:
        f.write(content)
    print("Patched _optimize_training_sample_weights!")
else:
    print("Could not find _optimize_training_sample_weights!")
