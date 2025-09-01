=== TRAINING PLACEHOLDER ANALYSIS SUMMARY ===

## Overview
The placeholder finder analyzed 211 files in src/training/ and found 7,282 placeholders.

## Breakdown by Type
- Pass statements: 3,626
- TODO comments: 3,653
- NotImplementedError raises: 0
- Placeholder functions: 3

## Directory Breakdown
  src/training: 64 files, 2320 placeholders
  src/training/core: 5 files, 174 placeholders
  src/training/examples: 1 files, 4 placeholders
  src/training/optimization: 9 files, 250 placeholders
  src/training/steps: 72 files, 3681 placeholders
  src/training/steps/analyst_training_components: 1 files, 20 placeholders
  src/training/steps/data_preparation_components: 1 files, 37 placeholders
  src/training/steps/multi_timeframe_training: 1 files, 64 placeholders
  src/training/steps/step1: 13 files, 336 placeholders
  src/training/steps/step17_final_parameters_optimization: 11 files, 351 placeholders
  src/training/steps/step4_analyst_labeling_feature_engineering_components: 4 files, 45 placeholders

PER FILE BREAKDOWN:
  src/training/__init__.py: 1 placeholders
    - todo_comments: 1

## Key Issues Found

### Most Problematic Areas:
1. **src/training/steps/**: 3,681 placeholders across 72 files
2. **src/training/**: 2,320 placeholders across 64 files
3. **src/training/optimization/**: 250 placeholders across 9 files

### Common Patterns:
- Empty exception handlers with TODO comments
- Pass statements in function bodies
- Unimplemented feature stubs

### Example Issues:

