"""Centralized defaults and constants for step03_5 and related modules."""

from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class Step03_5Defaults:
	# Reproducibility
	default_random_state: int = 42
	seed_deterministic: bool = True

	# KMeans
	kmeans_n_init: int = 10
	kmeans_max_iter: int = 100
	minibatch_n_init: int = 3

	# Cross-validation
	cv_n_splits: int = 5

	# Models
	rf_n_estimators_cv: int = 50
	rf_n_estimators: int = 100
	hgb_max_iter: int = 100
