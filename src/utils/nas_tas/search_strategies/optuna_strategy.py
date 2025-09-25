"""Optuna powered strategies including Bayesian and Hyperband variants."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

try:
    import optuna
except ImportError:  # pragma: no cover - optional dependency
    optuna = None  # type: ignore

from .base import Candidate, Evaluation, SearchState, SearchStrategy


class OptunaSearchStrategy(SearchStrategy):
    """Bayesian optimisation strategy backed by Optuna."""

    name = "optuna"

    def __init__(self, random_seed: Optional[int] = None, sampler: Optional[str] = None):
        super().__init__(random_seed=random_seed)
        self._sampler_name = sampler or "tpe"
        self._study: Optional["optuna.study.Study"] = None
        self.metric_names: Sequence[str] = []

    def initialize(
        self,
        search_space: Dict[str, Any],
        objective,
        state: SearchState,
        config: Dict[str, Any],
    ) -> None:  # type: ignore[override]
        if optuna is None:  # pragma: no cover - optional dependency
            raise ImportError(
                "Optuna is required for OptunaSearchStrategy. Install optuna to enable this strategy."
            )
        super().initialize(search_space, objective, state, config)
        directions = config.get("objective_directions")
        n_objectives = len(config.get("objective_names", []))
        if directions is None:
            directions = ["maximize"] * max(1, n_objectives)
        sampler = self._create_sampler()
        pruner = optuna.pruners.HyperbandPruner(
            min_resource=config.get("hyperband_min_resource", 1),
            max_resource=config.get("hyperband_max_resource", config.get("max_iterations", 100)),
        )
        self.metric_names = config.get("objective_names") or ["score"]
        study_name = config.get("study_name")
        self._study = optuna.create_study(
            directions=directions,
            study_name=study_name,
            sampler=sampler,
            pruner=pruner if config.get("enable_pruning", True) else None,
        )
        self.logger.info(
            "Created Optuna study",
            extra={
                "study_name": self._study.study_name,
                "directions": directions,
                "sampler": self._sampler_name,
                "objectives": list(self.metric_names),
            },
        )

    def _create_sampler(self) -> "optuna.samplers.BaseSampler":
        if self._sampler_name == "random":
            return optuna.samplers.RandomSampler(seed=self.random_seed)
        if self._sampler_name == "cmaes":
            return optuna.samplers.CmaEsSampler(seed=self.random_seed)
        return optuna.samplers.TPESampler(seed=self.random_seed, multivariate=True)

    def sample_candidates(
        self, state: SearchState, n_candidates: int
    ) -> List[Candidate]:
        assert self._study is not None
        candidates: List[Candidate] = []
        for _ in range(n_candidates):
            trial = self._study.ask()
            params = {}
            for name, spec in self.search_space.items():
                if isinstance(spec, dict):
                    param_type = spec.get("type", "float")
                    if param_type == "int":
                        params[name] = trial.suggest_int(name, spec.get("low", 0), spec.get("high", 10))
                    elif param_type == "float":
                        params[name] = trial.suggest_float(
                            name,
                            spec.get("low", 0.0),
                            spec.get("high", 1.0),
                            log=spec.get("log", False),
                        )
                    else:
                        params[name] = trial.suggest_categorical(name, spec.get("choices", []))
                elif isinstance(spec, list):
                    params[name] = trial.suggest_categorical(name, spec)
                else:
                    params[name] = spec
            candidates.append(Candidate(params=params, context=trial))
        return candidates

    def update_state(
        self, state: SearchState, evaluations: Sequence[Evaluation]
    ) -> None:
        assert self._study is not None
        for evaluation in evaluations:
            values = [evaluation.metrics.get(name, evaluation.score) for name in self.metric_names]
            trial = evaluation.candidate.context
            try:
                self._study.tell(trial, values if len(values) > 1 else values[0])
            except optuna.TrialPruned:
                self.logger.debug("Trial pruned", extra={"params": evaluation.candidate.params})
        state.register_evaluations(list(evaluations))
        state.iteration += 1
        if state.iteration >= self.config.get("max_iterations", 0):
            state.terminated = True
        if self.config.get("max_trials") and len(self._study.trials) >= self.config["max_trials"]:
            state.terminated = True

    def finalize(self, state: SearchState) -> Dict[str, Any]:
        result = super().finalize(state)
        if self._study is not None:
            result["study_name"] = self._study.study_name
            result["best_trial"] = self._study.best_trial.number if self._study.best_trial else None
            result["trials"] = [
                {
                    "number": trial.number,
                    "values": trial.values,
                    "params": dict(trial.params),
                    "state": str(trial.state),
                }
                for trial in self._study.trials
            ]
        return result


class HyperbandSearchStrategy(OptunaSearchStrategy):
    """Hyperband early stopping strategy built on top of Optuna."""

    name = "hyperband"

    def __init__(self, random_seed: Optional[int] = None):
        super().__init__(random_seed=random_seed, sampler="tpe")

    def initialize(
        self,
        search_space: Dict[str, Any],
        objective,
        state: SearchState,
        config: Dict[str, Any],
    ) -> None:  # type: ignore[override]
        config = dict(config)
        config.setdefault("enable_pruning", True)
        config.setdefault("objective_directions", ["maximize"])
        super().initialize(search_space, objective, state, config)

