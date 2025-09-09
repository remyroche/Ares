"""Seeding utilities for reproducible runs across common libraries."""

from __future__ import annotations

import os
import random
from typing import Optional

from .logger import system_logger

try:
	import numpy as np  # type: ignore
except Exception:  # pragma: no cover - numpy may not be installed in some envs
	np = None  # type: ignore


def seed_everything(seed: int = 42, deterministic: bool = True) -> None:
	"""
	Seed Python, NumPy, and optional ML libs for reproducibility.

	Args:
		seed: The seed value to use.
		deterministic: If True, set flags for deterministic operations when available.
	"""
	logger = system_logger.getChild("SeedUtils")
	try:
		# Python built-ins
		random.seed(seed)
		os.environ["PYTHONHASHSEED"] = str(seed)

		# NumPy
		if np is not None:
			np.random.seed(seed)

		# PyTorch (CPU/CUDA/MPS) - optional
		try:
			import torch  # type: ignore
			if hasattr(torch, "manual_seed"):
				torch.manual_seed(seed)
				if hasattr(torch, "cuda") and callable(getattr(torch.cuda, "manual_seed_all", None)):
					torch.cuda.manual_seed_all(seed)  # type: ignore[attr-defined]
				if deterministic:
					# Enable deterministic algorithms when possible
					try:
						import torch.backends.cudnn as cudnn  # type: ignore
						cudnn.deterministic = True  # type: ignore[attr-defined]
						cudnn.benchmark = False  # type: ignore[attr-defined]
					except Exception:
						pass
		except Exception:
			# Torch not available or seeding failed, continue
			pass

		logger.info(f"✅ Seeding initialized with seed={seed}, deterministic={deterministic}")
	except Exception as e:  # pragma: no cover
		logger.warning(f"⚠️ Seeding encountered an issue: {e}")

