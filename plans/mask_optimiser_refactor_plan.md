# Mask Optimiser Refactoring Plan

## Overview
Refactor `extreme_price_movements/mask_optimiser.py` (5787 lines) into a stage-based package structure while keeping the main entry point intact.

## Current Structure Analysis

### Phase/Stage Mapping
- **Stage 1 (Phase 1)**: Lines ~3771-3963 - Initial candidate evaluation on 50% symbols + 50% history with cheap metrics and primary classifier
- **Stage 2 (Phase 2)**: Lines ~4065-4208 - Full evaluation on all symbols & history with full metrics on top Phase 1 candidates
- **Stage 2.5**: Lines ~4570-4614 - Diversity filter / ranking between Phase 2 and Phase 3
- **Stage 3 (Phase 3)**: Lines ~4375-4500 - Feature learnability, conditional predictability, economic gain computation
- **Stage 4 (Phase 4)**: Lines ~4459+ - TBM LGBM metrics and final diagnostics

### Key Stage-Specific Functions
| Stage | Functions to Extract |
|-------|---------------------|
| Stage 1 | `_phase1_subsample_indices`, `_build_phase_local_shared`, `_compute_primary_phase1_classifier_gain` |
| Stage 2 | Candidate grid evaluation with full metrics |
| Stage 2.5 | Diversity filtering and ranking |
| Stage 3 | `_compute_phase3_feature_learnability`, `_compute_conditional_predictability_metrics`, `_compute_tbm_economic_gain`, `_compute_mfe_coverage` |
| Stage 4 | `_compute_phase4_tbm_lgbm_metrics`, `_final_topk_diagnostics` |

### Shared Utilities (to remain accessible)
- **Numba kernels**: rolling_max_index_nb, rolling_min_index_nb, rolling_std_nb, dilate_mask_by_groups_nb, tbm_outcomes_atr_nb, compute_impulse_coherence_nb, etc.
- **Safe Python equivalents**: rolling_max_index_safe, rolling_min_index_safe, etc.
- **Mode helpers**: _mode_is_up, _mode_is_tf, _get_side_mask, _mode_primary_target, _signed_mode_return
- **Data helpers**: _build_temporal_folds, _build_day_ids, _build_timestamp_ids, etc.
- **Metric helpers**: _zscore_np, _log_stage_snapshot, _compute_full_metrics_for_candidate
- **Constants**: ALL_MODES, MODE_PRICE_UP_TF, etc.

## Target Package Structure

```
extreme_price_movements/
├── mask_optimiser.py          # Main orchestrator (refactored to import from package)
└── mask_optimiser/            # New package directory
    ├── __init__.py            # Package exports
    ├── mask_optimiser_1.py    # Stage 1 logic
    ├── mask_optimiser_2.py    # Stage 2 logic
    ├── mask_optimiser_2_5.py   # Stage 2.5 logic  
    ├── mask_optimiser_3.py    # Stage 3 logic
    ├── mask_optimiser_4.py    # Stage 4 logic
    └── shared.py              # Shared utilities used across stages
```

## Refactoring Steps

1. **Create package directory** `extreme_price_movements/mask_optimiser/`

2. **Create `shared.py`** - Move all functions used by multiple stages:
   - Numba kernels (lines 112-381)
   - Safe Python equivalents (lines 384-627)
   - Mode helper functions (lines 664-780)
   - Data building functions (lines 879-1100)
   - Common metric functions
   - Constants and configuration helpers

3. **Create stage modules**:
   - `mask_optimiser_1.py` - Phase 1 candidate evaluation
   - `mask_optimiser_2.py` - Phase 2 full evaluation  
   - `mask_optimiser_2_5.py` - Diversity filter
   - `mask_optimiser_3.py` - Phase 3 feature learnability + economics
   - `mask_optimiser_4.py` - Phase 4 TBM metrics + diagnostics

4. **Refactor `mask_optimiser.py`**:
   - Keep CLI entry point (`run_mask_optimization_4modes`)
   - Keep `optimize_layer0_masks_by_mode` and `optimize_layer_masks_by_mode` as orchestrators
   - Import stage functions from the new package
   - Import shared utilities from the package

5. **Update imports** in dependent files:
   - `position_sizer_v2.py`
   - `test_mask_optimiser.py`
   - Any other files importing from mask_optimiser

## Key Design Decisions

- **Function signatures**: Preserve exact signatures for backward compatibility
- **Caching behavior**: Maintain existing caching mechanisms within stages
- **Output artifacts**: Preserve all CSV/JSON outputs with same naming
- **CLI interface**: Keep existing argparse interface unchanged

---

## Detailed Function Mappings

### Stage 1 (mask_optimiser_1.py) - Phase 1: Initial Candidate Filtering
**Location in source**: Lines ~3771-3963 (inside `_run_mode_search`)

**Functions to extract**:
- `_phase1_subsample_indices(shared, cfg, seed=42)` → Line 2804
- `_build_phase_local_shared(shared, phase1_mask)` → Line 2858
- `_compute_primary_phase1_classifier_gain(mode, side_mask, ...)` → Line 2899

**Stage entry function**:
```python
def run_stage1_phase1(
    shared: Dict[str, Any],
    feature_dict: Dict[str, np.ndarray],
    cfg: Dict[str, Any],
    candidate_grid: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Stage 1: Evaluate candidates on 50% symbols + 50% history.
    
    Returns list of phase1_rows with candidate stats.
    """
```

**Dependencies from shared**:
- `_phase1_subsample_indices`
- `_build_phase_local_shared`
- `_compute_z_cache` (also used by Stage 2)
- `_generate_event_masks_fast`
- `_get_side_mask`
- `_mode_primary_target`
- `_signed_mode_return`

---

### Stage 2 (mask_optimiser_2.py) - Phase 2: Full Metric Evaluation
**Location in source**: Lines ~4065-4208

**Functions to extract**: (inline code in `_run_mode_search`)

**Stage entry function**:
```python
def run_stage2_phase2(
    shared: Dict[str, Any],
    feature_dict: Dict[str, np.ndarray],
    cfg: Dict[str, Any],
    phase1_rows: List[Dict[str, Any]],
    candidate_registry: Dict[str, Dict[str, Any]],
    global_z_cache: Dict,
) -> pd.DataFrame:
    """
    Stage 2: Full evaluation on all symbols & history.
    
    Returns DataFrame with full metrics for filtered candidates.
    """
```

**Key computations**:
- `_coherence_metrics_single_side`
- `_compute_regime_distinctness_single_side`
- `_compute_full_metrics_for_candidate`
- `_compute_legacy_conditional_learnability`

---

### Stage 2.5 (mask_optimiser_2_5.py) - Diversity Filtering & Ranking
**Location in source**: Lines ~4570-4614

**Functions to extract**: (inline code in `_run_mode_search`)

**Stage entry function**:
```python
def run_stage2_5_diversity_filter(
    df2: pd.DataFrame,
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    """
    Stage 2.5: Apply diversity filter and ranking.
    
    - Global top + at least 1 stable per family
    - Max 3 per family
    - Compute D_r, N_r, S_r scores
    """
```

---

### Stage 3 (mask_optimiser_3.py) - Phase 3: Feature Learnability & Economics
**Location in source**: Lines ~4375-4500 (runs after Stage 2.5)

**Functions to extract**:
- `_compute_phase3_feature_learnability(shared, feature_dict, side_mask, mode, folds, cfg)` → Line 2929
- `_compute_conditional_predictability_metrics(shared, side_mask, mode, folds, cfg)` → Line 3003
- `_compute_tbm_economic_gain(shared, side_mask, mode, folds, cfg)` → Line 3111
- `_compute_mfe_coverage(shared, side_mask, cfg)` → Line 3414

**Stage entry function**:
```python
def run_stage3_phase3(
    shared: Dict[str, Any],
    feature_dict: Dict[str, np.ndarray],
    cfg: Dict[str, Any],
    df2: pd.DataFrame,
    candidate_registry: Dict[str, Dict[str, Any]],
    folds: List[Tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    """
    Stage 3: Feature learnability, conditional predictability, economic gain.
    
    Returns DataFrame with all Phase 3 metrics merged.
    """
```

---

### Stage 4 (mask_optimiser_4.py) - Phase 4: TBM LGBM & Final Diagnostics
**Location in source**: Lines ~5283-5400

**Functions to extract**:
- `_compute_phase4_tbm_lgbm_metrics(shared, side_mask, folds, cfg, per_geometry_metrics)` → Line 3270
- `_final_topk_diagnostics(mode, df_diag_input, candidate_masks, shared, feature_dict, cfg)` → Line 3623

**Stage entry function**:
```python
def run_stage4_phase4(
    shared: Dict[str, Any],
    feature_dict: Dict[str, np.ndarray],
    cfg: Dict[str, Any],
    df3: pd.DataFrame,
    candidate_masks: Dict[str, Dict[str, np.ndarray]],
    folds: List[Tuple[np.ndarray, np.ndarray]],
) -> Dict[str, Any]:
    """
    Stage 4: TBM LGBM metrics and final diagnostics.
    
    Returns dict with best config, shortlist, diagnostics.
    """
```

---

## Shared Utilities Module (shared.py)

### Numba Kernels (Lines 112-381)
| Function | Lines | Purpose |
|----------|-------|---------|
| `rolling_max_index_nb` | 112-146 | Rolling max with index |
| `rolling_min_index_nb` | 149-183 | Rolling min with index |
| `rolling_std_nb` | 186-216 | Rolling standard deviation |
| `dilate_mask_by_groups_nb` | 219-234 | Dilate mask by group |
| `tbm_outcomes_atr_nb` | 237-288 | TBM outcome calculation |
| `compute_impulse_coherence_nb` | 304-381 | Impulse coherence metrics |
| `active_days_fraction_nb` | 529-541 | Active days fraction |
| `daily_event_stats_nb` | 543-568 | Daily event statistics |
| `fold_base_rate_nb` | 570-584 | Fold base rate |
| `simple_mask_count_nb` | 586-587 | Simple mask count |
| `_rolling_robust_z_1d` | 2205-2227 | Robust z-score rolling |

### Safe Python Equivalents (Lines 384-627)
| Function | Lines | Purpose |
|----------|-------|---------|
| `rolling_max_index_safe` | 384-400 | Safe rolling max |
| `rolling_min_index_safe` | 403-419 | Safe rolling min |
| `rolling_std_safe` | 422-435 | Safe rolling std |
| `compute_impulse_coherence_safe` | 438-510 | Safe impulse coherence |
| `dilate_mask_by_asset_safe` | 513-527 | Safe mask dilation |
| `active_days_fraction_safe` | 591-600 | Safe active days |
| `daily_event_stats_safe` | 602-612 | Safe daily stats |
| `fold_base_rate_safe` | 614-623 | Safe fold rate |
| `simple_mask_count_safe` | 625-626 | Safe mask count |

### Mode Helper Functions (Lines 664-780)
| Function | Lines | Purpose |
|----------|-------|---------|
| `_mode_is_up` | 664-665 | Check if mode is up |
| `_mode_is_tf` | 668-669 | Check if mode is tf |
| `_get_side_mask` | 672-673 | Get side mask |
| `_mode_primary_target` | 676-694 | Get primary target |
| `_signed_mode_return` | 696-710 | Get signed returns |
| `_dedup_universe_by_base` | 80-104 | Deduplicate symbols |

### Data Building Functions (Lines 879-2191)
| Function | Lines | Purpose |
|----------|-------|---------|
| `_build_temporal_folds` | 879-984 | Build CV folds |
| `_build_day_ids` | 2044-2047 | Build day IDs |
| `_build_timestamp_ids` | 2050-2052 | Build timestamp IDs |
| `_build_vol_regime_ids` | 2055-2066 | Build vol regime IDs |
| `_build_candidate_grid` | 2127-2188 | Build candidate grid |
| `_build_asset_groups_from_codes` | 2192-2203 | Build asset groups |

### Metric Functions (Multiple locations)
| Function | Lines | Purpose |
|----------|-------|---------|
| `_zscore_np` | 751-759 | Z-score normalization |
| `_metric_or_nan` | 762-769 | Metric or NaN |
| `_safe_abs_ratio` | 772-778 | Safe ratio |
| `_log_stage_snapshot` | 780-795 | Log stage results |
| `_coherence_metrics_single_side` | 797-819 | Coherence metrics |
| `_compute_regime_distinctness_single_side` | 822-874 | Regime distinctness |
| `_compute_full_metrics_for_candidate` | 3429-3619 | Full candidate metrics |
| `_compute_legacy_conditional_learnability` | 1657-1758 | Legacy learnability |

### Constants (Lines 59-77)
```python
MODE_PRICE_UP_TF = "price_up_tf"
MODE_PRICE_UP_MR = "price_up_mr"
MODE_PRICE_DOWN_TF = "price_down_tf"
MODE_PRICE_DOWN_MR = "price_down_mr"
ALL_MODES = [MODE_PRICE_UP_MR, MODE_PRICE_UP_TF, MODE_PRICE_DOWN_MR, MODE_PRICE_DOWN_TF]
```

## Import Dependencies Between Stages

```
                      ┌─────────────────┐
                      │    shared.py    │
                      │  (Numba kernels │
                      │   + helpers)    │
                      └────────┬────────┘
                               │
        ┌──────────┬───────────┼───────────┬──────────┐
        │          │           │           │          │
        ▼          ▼           ▼           ▼          ▼
   ┌─────────┐ ┌─────────┐ ┌──────────┐ ┌─────────┐ ┌─────────┐
   │Stage 1  │ │Stage 2  │ │Stage 2.5 │ │Stage 3  │ │Stage 4  │
   │ Phase1 │ │ Phase2  │ │Diversity │ │ Phase3  │ │ Phase4  │
   └────┬────┘ └────┬────┘ └────┬─────┘ └────┬────┘ └────┬────┘
        │           │            │            │           │
        └──────────┴────────────┴────────────┴───────────┘
                               │
                               ▼
                      ┌─────────────────┐
                      │ mask_optimiser.py│
                      │   (orchestrator) │
                      └─────────────────┘
```

## Function Signatures to Preserve

All function signatures must remain unchanged for backward compatibility:

### Stage Entry Functions (new, with clear signatures)
```python
# Stage 1
def run_stage1_phase1(
    shared: Dict[str, Any],
    feature_dict: Dict[str, np.ndarray],
    cfg: Dict[str, Any],
    candidate_grid: List[Dict[str, Any]],
) -> List[Dict[str, Any]]

# Stage 2  
def run_stage2_phase2(
    shared: Dict[str, Any],
    feature_dict: Dict[str, np.ndarray],
    cfg: Dict[str, Any],
    phase1_rows: List[Dict[str, Any]],
    candidate_registry: Dict[str, Dict[str, Any]],
    global_z_cache: Dict,
) -> pd.DataFrame

# Stage 2.5
def run_stage2_5_diversity_filter(
    df2: pd.DataFrame,
    cfg: Dict[str, Any],
) -> pd.DataFrame

# Stage 3
def run_stage3_phase3(
    shared: Dict[str, Any],
    feature_dict: Dict[str, np.ndarray],
    cfg: Dict[str, Any],
    df2: pd.DataFrame,
    candidate_registry: Dict[str, Dict[str, Any]],
    folds: List[Tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame

# Stage 4
def run_stage4_phase4(
    shared: Dict[str, Any],
    feature_dict: Dict[str, np.ndarray],
    cfg: Dict[str, Any],
    df3: pd.DataFrame,
    candidate_masks: Dict[str, Dict[str, np.ndarray]],
    folds: List[Tuple[np.ndarray, np.ndarray]],
) -> Dict[str, Any]
```

## Verification Checklist

After refactoring, verify:

1. **Entry Point Preserved**
   - [ ] `python -m extreme_price_movements.mask_optimiser --help` works
   - [ ] CLI arguments unchanged

2. **Stage Modules**
   - [ ] Each stage in its own module
   - [ ] Clear entry function per stage
   - [ ] No circular imports

3. **Functionality**
   - [ ] End-to-end run produces same outputs
   - [ ] Caching behavior unchanged
   - [ ] Candidate naming preserved
   - [ ] CSV outputs identical

4. **Backward Compatibility**
   - [ ] External imports still work:
     - `from extreme_price_movements.mask_optimiser import optimize_layer0_masks_by_mode`
     - `from extreme_price_movements.mask_optimiser import _mode_primary_target`
     - `from extreme_price_movements.mask_optimiser import _compute_regime_distinctness`

5. **Shared Utilities**
   - [ ] All Numba kernels accessible
   - [ ] All helper functions accessible
   - [ ] Constants exported

## Implementation Order

1. Create `extreme_price_movements/mask_optimiser/` directory
2. Create `__init__.py` with re-exports
3. Create `shared.py` with all shared utilities
4. Create `mask_optimiser_1.py` - Stage 1
5. Create `mask_optimiser_2.py` - Stage 2
6. Create `mask_optimiser_2_5.py` - Stage 2.5
7. Create `mask_optimiser_3.py` - Stage 3
8. Create `mask_optimiser_4.py` - Stage 4
9. Refactor `mask_optimiser.py` to use new package
10. Verify all imports and run end-to-end test
