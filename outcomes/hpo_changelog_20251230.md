# HPO Pipeline Changes Log - 2025-12-30

## Session Overview
Running `meta_labeling_hpo_sample_weighted` with:
- Symbol: ETHUSDT
- Execution Mode: blank
- Flags: --force-hpo --enable-labeling-hpo

## Changes Made

### [Change #1] - 19:43
- **Script**: `src/training/steps/labeling/label_based_layer_2.py`
- **Function**: Module-level import
- **Change Description**: Made `catboost` import optional by wrapping in try/except with `CATBOOST_AVAILABLE` flag
- **Reason**: Module was failing to load due to missing catboost package, preventing `meta_labeling_hpo_sample_weighted` step from registering
- **Line Numbers**: 22-27

### [Change #2] - 20:03
- **Script**: `src/training/steps/labeling/__init__.py`
- **Function**: Module-level import block
- **Change Description**: Removed erroneous string literal `"GlobalMetaLabelingHPOSampleWeightedStep"` from `winning_feature_set_selector` import statement
- **Reason**: Syntax error was blocking the entire labeling module from loading, preventing step registration
- **Line Numbers**: 126-134

### [Change #3] - 20:32
- **Script**: `src/training/steps/labeling/orthogonal_label_generation.py`
- **Function**: `adaptive_threshold_calculator`
- **Change Description**: Used forward reference string annotation for `BaseEventGenerator` type hint (changed `generator: BaseEventGenerator` to `generator: "BaseEventGenerator"`)
- **Reason**: Class was used in type hint before its definition, causing `NameError: name 'BaseEventGenerator' is not defined` during module import
- **Line Numbers**: 555

### [Change #4] - 20:36
- **Script**: `src/launcher/ares_launcher.py`
- **Function**: `SimplifiedAresLauncher.__init__`, `step_registry` property, `run_step()`
- **Change Description**: Made `step_registry` import lazy by:
  1. Removing module-level `from src.training.steps.base_step import step_registry, BaseStep` (line 65)
  2. Converting `self.step_registry` to lazy-loaded property
  3. Changed `run_step()` to always call `import_step_package_for_step(step_name)` before accessing registry
- **Reason**: Module-level import triggered chain imports through `utils/__init__.py`, loading 478 feature generators, VectorBT, GPU tests, etc. on every launcher startup (~7-10 min)
- **Line Numbers**: 64-72, 248-260, 276-285

---

### [Change #5] - 21:15
- **Script**: `src/training/steps/labeling/orthogonal_label_generation.py`
- **Function**: Module-level (Section 7 at end of file)
- **Change Description**: Restored full implementations (not aliases) from git history:
  - `generate_dual_cusum_signals` - Kalman-based dual CUSUM for trend vs reversal
  - `AdaptiveSymmetricCUSUMEvents` - Dynamic threshold CUSUM based on volatility
  - `SymmetricCusumEvents` - Simple fixed-threshold CUSUM
  - `ImprovedCUSUMEvents` - CUSUM with differentiated trend/reversal weights (w_trend, w_reversal params)
  - `VolatilityShockEvents` - Detects vol shocks via z-score
  - `TrendInitiationEvents` - MA crossover detection
  - `MeanReversionExtremeEvents` - Z-score based MR extremes
  - `LiquidityShockEvents` - Volume z-score based
  - `TimeEvents` - Periodic time-based events
- **Reason**: `label_based_layer_2.py` imports these classes. Originally used simple aliases but per user feedback, restored actual de Prado-style implementations with differentiated CUSUM for trend vs mean reversion
- **Line Numbers**: 1005-1230


---

### [Change #6] - 22:15
- **Script**: `src/feature_generation/core/feature_bank.py`
- **Function**: `_auto_register_generators`, `get_generators_by_category`, `__init__`
- **Change Description**: Implemented lazy loading for feature generators.
  - Replaced eager instantiation loop with `_lazy_registry` population.
  - Added `_ensure_category_loaded` for on-demand instantiation.
  - Ensured Singleton instance correctly propagates `_lazy_registry`.
- **Reason**: Performance optimization. Pipeline startup time was excessive (~10m) due to eager creation of 400+ generators. Lazy loading defers creation until the category is actually requested by a pipeline layer.
- **Line Numbers**: 206, 210, 257-384

---

## Change Entry Template
### [Change #X] - [HH:MM]
- **Script**: 
- **Function**: 
- **Change Description**: 
- **Reason**: 
- **Line Numbers**: 

---

### [Change #7] - 23:35
- **Script**: `src/training/steps/labeling/orthogonal_label_generation.py`
- **Function**: `check_label_quality`, `orthogonal_label_generation`
- **Change Description**: Fixed and Enhanced Signal Generation:
  1. **Fixed Parameter Passing**: Refactored `orthogonal_label_generation` to pass adaptive parameters as kwargs instead of positional args, allowing `generate_adaptive` to correctly invoke `_adjust_z_threshold`.
  2. **Fixed NameError**: Corrected variable usage in `check_label_quality` (restored n/rate calculation).
  3. **Fixed Return Type**: Changed fallback return from `([], {})` to `[]` to match expected type in Layer 2.
  4. **Enhanced Logging**: Implemented comprehensive CSV logging of all candidate trials (rejection reasons, metrics) saved to `outcomes/geometry_gates_*.csv`.
  5. **Logic Fix**: Moved CSV save logic to run before "No candidates passed" early exit to ensure logs are preserved.
- **Reason**: 
  - Adaptive thresholding was failing to adjust parameters, leading to low signal rates (0.26/day vs 7.5 target).
  - Lack of visibility into rejection reasons hindered debugging. 
  - "Sample Size" gate failures (0.88 < 1.0) now fully observable via CSV.
- **Line Numbers**: 360-380, 1830-1845, 1918-1930, 1945-1955

### [Change #8] - 23:55
- **Script**: `src/training/steps/labeling/orthogonal_label_generation.py`
- **Function**: `check_label_quality`, `generate_adaptive`, `_adjust_z_threshold`
- **Change Description**:
  1. **Disabled Gate**: Converted "Perturbation Stability" check from a hard failure to a warning (lines 427-429).
  2. **Enhanced Adaptive Logic**: 
     - Increased `max_iterations` to 20.
     - Implemented "Panic Mode": aggressively relaxes parameters by 50% if signal rate < 10% of target.
     - Fixed **Inverted Logic Bug**: `factor` was increasing (tightening) when signals were low; inverted to decrease (relax) correctly.
     - Removed early exit on 0 initial events to allow adaptive recovery.
  3. **Multi-Param Relaxation**: Updated `_adjust_z_threshold` for all generator families to relax secondary constraints (`volume`, `min_move`, `lookback`) alongside `z-score`.
- **Reason**: 
  - User requested removal of Perturbation gate.
  - Signal rates were critically low (<1.0/day).
  - Adaptive logic bug was preventing recovery from low/zero signal counts.
- **Line Numbers**: 427-430, 638-685, 804-970, 1000-1630

### [Change #9] - 00:03
- **Script**: `src/training/steps/labeling/orthogonal_label_generation.py`
- **Function**: `check_label_quality`
- **Change Description**: Manually relaxed additional gates:
  - **Class Balance**: Relaxed limits from [0.075, 0.925] to [0.05, 0.95].
  - **PSR**: Relaxed minimum PSR threshold from 0.90 to 0.85.
- **Reason**: Manual intervention by user to unblock pipeline flow by widening acceptance criteria.
- **Line Numbers**: 382, 457

---

### [Change #10] - 00:35
- **Script**: `src/training/steps/labeling/orthogonal_label_generation.py`
- **Function**: `run_lgbm_probe`
- **Change Description**: Implemented **Tailored Focal Loss** for the LGBM Probe.
  - Added `focal_loss_utils.py` with `get_focal_loss_lgbm`.
  - Probe now calculates dynamic `alpha = 1 - pos_rate` for each candidate geometry.
  - Uses custom Focal Loss objective instead of standard binary logloss/AUC.
- **Reason**: To improve probe sensitivity and signal detection for imbalanced geometries, matching the "strong" model behavior in Layer 2.
- **Line Numbers**: 606-620
