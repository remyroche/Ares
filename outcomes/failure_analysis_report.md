# Layer 2 Geometry Generation Failure Analysis
**Date:** 2025-12-30 23:50
**Run ID:** 20251230_234813
**Status:** CRITICAL FAILURE (0 Passing Candidates)

## Executive Summary
The `orthogonal_label_generation` step produced **972** candidate configurations across **8** generator families. 
**Zero (0)** candidates passed the quality gates.

## Failure Statistics by Family
All families failed to produce valid geometries.

| Family | Total Trials | Passing | Failure Rate |
|---|---|---|---|
| BREAKOUT | 121 | 0 | 100% |
| ENTROPY | 121 | 0 | 100% |
| KALMAN_REGIME | 121 | 0 | 100% |
| KALMAN_TREND | 121 | 0 | 100% |
| MR_VWAP | 121 | 0 | 100% |
| ORDER_BLOCKS | 121 | 0 | 100% |
| SR_BREAKOUTS | 124 | 0 | 100% |
| VWAP_CROSS | 121 | 0 | 100% |

## Rejection Reason Breakdown
The following gates were responsible for rejections (highest to lowest):

1.  **Perturbation Stability (Jaccard < 0.5)**: **339** failures
    *   *Observation*: High instability in generated labels when noise is added.
2.  **Sample Size (< 1.0/day)**: **216** failures
    *   *Observation*: Generators failing to produce enough signals even with adaptive thresholding.
3.  **Class Balance (< 7.5% or > 92.5%)**: **213** failures
    *   *Observation*: Labels are either too sparse (mostly 0) or too frequent.
4.  **ANOVA (p-value > 0.05)**: **134** failures
    *   *Observation*: No statistically significant difference in returns between labeled and unlabeled data.
5.  **Mutual Info**: **56** failures
6.  **PSR (Probabilistic Sharpe Ratio)**: **14** failures

## Logs Location
Full logs available at: `outcomes/geometry_gates_20251230_234813.csv`
