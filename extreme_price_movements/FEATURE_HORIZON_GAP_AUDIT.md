# Feature Horizon Gap Audit (2h / 4h / 8h)

## Scope
This audit checks whether feature coverage for the base model families (TF, MR, long, short) is properly adapted to the three traded horizons configured in the system: **2h, 4h, 8h**.

## What is already present
- Labels are horizon-aware: `label_horizons_hours = [2, 4, 8]` with per-horizon TP/SL arrays.
- Core return ladder includes `ret2h`, `ret4h`, `ret8h` (plus other windows).
- Several multi-horizon structure features exist: `ft_2/4/8`, `pullback_2/4/8`, `donch_dist_2/4/8`.
- TF/MR feature lists include many of the above and are shared across long/short direction models.

## Gaps that directly impact 2h / 4h / 8h trading

### 1) Missing short-horizon risk normalization for 2h/4h
- Volatility features currently include `rv_6h`, `rv_8h`, `rv_12h`, `rv_24h` but **not** `rv_2h` or `rv_4h`.
- This weakens calibration for fast entries/exits and makes 2h/4h signals rely on longer-vol proxies.

### 2) Missing 2h path-quality features
- Path/outcome quality exists only as `mfe_4h`, `mae_4h`, `mfe_8h`, `mae_8h`.
- There is no `mfe_2h` / `mae_2h`, so the 2h horizon cannot learn immediate excursion/asymmetry patterns with dedicated features.

### 3) Several "new" features are conditionally blocked and likely never materialize
- `rsi_lag1` and `rsi_1h_slope` are computed only if `"rsi" in feats` at a point where `rsi` has not yet been added.
- `clv_mean_24` is computed only if `"clv" in feats` before `clv` is created.
- `atr_pct_change` is computed only if `"atr_pct" in feats` before `atr_pct` is created.
- Effect: configured features can be expected by models but absent in the actual feature frame.

### 4) Config helper feature list omits some horizon primitives that are computed
- `HELPER_BASE_FEATURES` includes `ret1h`, `ret6h`, `ret8h` but omits `ret2h` and `ret4h` despite these being generated.
- This reduces candidate breadth specifically for 2h/4h heads when selection starts from configured keys.

### 5) Regime context is mostly 24h-anchored, not horizon-parallel
- Market regime features rely heavily on `mkt_ret24h`, `mkt_rv` (24h std), and 6h aggregate.
- There are no explicit `mkt_ret2h/4h/8h` or `mkt_rv_2h/4h/8h` variants to align regime detection with each trading horizon.

### 6) Many mature features are tied to fixed windows that are not mapped to 2/4/8
- Examples: `breakout_24h`, `jump_rate_10h`, `accel_5h`, `volume_price_corr_10h`, and several 24h smoothers.
- These can still be useful globally, but without horizon-specific counterparts they bias representation toward slower dynamics.

## Base-model specific implications

### TF
- Strength: has `ft_2/4/8`, `pullback_2/4/8`, `breakout_24h`, coherence and flow proxies.
- Missing for effective 2h/4h TF: fast risk-normalizers (`rv_2h/rv_4h`) and horizon-matched market regime (`mkt_ret2h/4h`, `mkt_rv_2h/4h`).

### MR
- Strength: has pullback and failure progression across 2/4/8, plus 4h/8h MFE/MAE.
- Missing: 2h excursion features (`mfe_2h`, `mae_2h`) and reliable availability of `rsi_1h_slope`, `atr_pct_change`, `clv_mean_24`.

### Long / Short directional models
- Current setup uses shared TF/MR feature pools for both directions.
- For horizon execution quality, directional asymmetry is only partially captured and not explicitly horizon-expanded.
- Practical gap: no dedicated 2h directional path-risk block (e.g., side-aware excursion + volatility + liquidity at 2h).

## Minimum additions to trade all three horizons effectively
1. Add `rv_2h` and `rv_4h` and expose them to TF/MR/meta key sets.
2. Add `mfe_2h` and `mae_2h` (matching current 4h/8h construction style).
3. Fix feature build order so `rsi_lag1`, `rsi_1h_slope`, `clv_mean_24`, `atr_pct_change` are always computed when configured.
4. Include `ret2h` and `ret4h` in helper/base selectable keys.
5. Add horizon-parallel market regime block: `mkt_ret2h`, `mkt_ret4h`, `mkt_ret8h`, `mkt_rv_2h`, `mkt_rv_4h`, `mkt_rv_8h`.
6. Add horizon variants for key structure features currently fixed at non-target windows (starting with breakout/accel/correlation windows).

## Priority order
1. **Correctness first**: fix missing-by-order features.
2. **2h viability**: add 2h volatility + excursion features.
3. **4h calibration**: add 4h volatility + regime variants.
4. **Consistency layer**: horizon-parallel market/regime and selected window variants.
