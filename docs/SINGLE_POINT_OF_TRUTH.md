# Single Point of Truth for Regime Merging Parameters

## Overview

This document establishes the single points of truth for all regime merging parameters to ensure consistency across the codebase.

## Parameter Definitions

### 1. **min_frequency**
**Single Point of Truth**: `REGIME_MERGING_CONFIG["min_frequency"] = 0.003`

**Purpose**: Minimum frequency threshold (0.3%) that determines which regimes are considered "low-frequency" and eligible for merging.

**All References Now Consistent**:
- Configuration: `0.003`
- Function default: `0.003`
- Command line default: `0.003`
- Fallback: `0.003`

### 2. **similarity_threshold**
**Single Point of Truth**: `REGIME_MERGING_CONFIG["similarity_threshold"] = 0.8`

**Purpose**: Threshold for regime similarity (80%) - regimes with similarity above this threshold can be merged.

**All References Now Consistent**:
- Configuration: `0.8`
- Function default: `0.8`
- Command line default: `0.8`
- Fallback: `0.8`

### 3. **max_regimes**
**Single Point of Truth**: `REGIME_MERGING_CONFIG["max_regimes"] = 20`

**Purpose**: Maximum number of regimes after merging (target 20 for 70-80% concentration).

**All References Now Consistent**:
- Configuration: `20`
- Function default: `20`
- Command line default: `20`
- Fallback: `20`

### 4. **target_top_20_concentration**
**Single Point of Truth**: `REGIME_MERGING_CONFIG["target_top_20_concentration"] = 0.80`

**Purpose**: Target concentration (80%) in the top 20 regimes.

**All References Now Consistent**:
- Configuration: `0.80`
- Function default: `0.80`
- Command line default: `0.80`
- Fallback: `0.80`

### 5. **aggressive_merging**
**Single Point of Truth**: `REGIME_MERGING_CONFIG["aggressive_merging"] = True`

**Purpose**: Enable aggressive merging strategies for higher concentration.

**All References Now Consistent**:
- Configuration: `True`
- Function default: `True`
- Command line default: `True`
- Fallback: `True`

## Configuration Block

```python
REGIME_MERGING_CONFIG = {
    "min_frequency": 0.003,          # 0.3% minimum frequency to keep regime separate
    "similarity_threshold": 0.8,     # 80% similarity threshold for merging
    "max_regimes": 20,               # Maximum total regimes after merging
    "enable_merging": True,          # Enable regime merging
    "merge_strategy": "similarity",   # "similarity" or "frequency"
    "target_top_20_concentration": 0.80,  # Target 80% concentration in top 20 regimes
    "aggressive_merging": True,      # Enable aggressive merging strategies
}
```

## Function Signature

```python
def merge_similar_regimes(
    cluster_df: pd.DataFrame,
    cluster_centroids: Dict[int, np.ndarray],
    similarity_threshold: float = 0.8,  # Match configuration
    min_frequency: float = 0.003,        # Match configuration
    max_regimes: int = 20,               # Match configuration
    target_top_20_concentration: float = 0.80,  # Match configuration
    aggressive_merging: bool = True,     # Match configuration
) -> Tuple[pd.DataFrame, Dict[int, int]]:
```

## Command Line Interface

```bash
python src/training/steps/step1_7_hmm_regime_discovery.py \
  --similarity-threshold 0.8 \     # Default: 0.8
  --min-frequency 0.003 \          # Default: 0.003
  --max-regimes 20 \               # Default: 20
  --target-concentration 0.80 \    # Default: 0.80
  --aggressive-merging             # Default: True
```

## Usage Guidelines

### For 70% Concentration
```python
REGIME_MERGING_CONFIG.update({
    "min_frequency": 0.005,          # 0.5%
    "similarity_threshold": 0.80,     # 80%
    "max_regimes": 25,               # 25 regimes
    "target_top_20_concentration": 0.70,
})
```

### For 80% Concentration (Recommended)
```python
# Use default configuration - already optimized for 80%
```

### For 80% Concentration
```python
REGIME_MERGING_CONFIG.update({
    "min_frequency": 0.002,          # 0.2%
    "similarity_threshold": 0.70,     # 70%
    "max_regimes": 15,               # 15 regimes
    "target_top_20_concentration": 0.80,
})
```

## Maintenance

When modifying these parameters:

1. **Always update the configuration first**: `REGIME_MERGING_CONFIG`
2. **Update function defaults** to match configuration
3. **Update command line defaults** to match configuration
4. **Update fallback values** to match configuration
5. **Update documentation** to reflect changes
6. **Test all entry points** to ensure consistency

## Validation

To verify single points of truth are maintained:

```python
# Check configuration
print(REGIME_MERGING_CONFIG)

# Check function defaults (inspect function signature)
import inspect
sig = inspect.signature(merge_similar_regimes)
print(sig.parameters)

# Check command line defaults
import argparse
parser = argparse.ArgumentParser()
# ... add arguments ...
args = parser.parse_args([])  # Empty args to get defaults
print(vars(args))
```

This ensures all parameters have a single, consistent source of truth across the entire codebase.
