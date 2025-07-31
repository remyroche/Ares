import logging
import pandas as pd
from datetime import timedelta


class DynamicWeighter:
    """
    ## CHANGE: Implemented the new Dynamic Weighter.
    ## This class is responsible for analyzing the recent performance of each base
    ## model within an ensemble and adjusting their weights to favor those that
    ## are performing well in current market conditions.
    """

    def __init__(self, config):
        self.config = config.get("supervisor", {}).get("dynamic_weighter", {})
        self.logger = logging.getLogger(self.__class__.__name__)
        self.max_adjustment = self.config.get("max_adjustment", 0.30)
        self.missed_opportunity_penalty = self.config.get(
            "missed_opportunity_penalty", -0.5
        )
        self.incorrect_penalty = self.config.get("incorrect_penalty", -1.5)
        self.correct_reward = self.config.get("correct_reward", 1.0)

    def run_daily_adjustment(
        self, ensemble_orchestrator, prediction_history: pd.DataFrame
    ):
        """
        Main method to orchestrate the daily weight adjustment for all ensembles.

        Args:
            ensemble_orchestrator: The main orchestrator object holding all ensembles.
            prediction_history: A DataFrame with recent prediction data.
        """
        self.logger.info("Starting daily dynamic weight adjustment...")

        end_date = pd.Timestamp.now(tz="UTC")
        start_date = end_date - timedelta(days=7)
        recent_history = prediction_history[
            prediction_history["timestamp"] >= start_date
        ]

        if recent_history.empty:
            self.logger.warning(
                "No recent prediction history found. Skipping weight adjustment."
            )
            return

        for regime, ensemble in ensemble_orchestrator.regime_ensembles.items():
            self.logger.info(f"Adjusting weights for {regime} ensemble...")
            ensemble_history = recent_history[recent_history["regime"] == regime]
            if not ensemble_history.empty:
                self._adjust_ensemble_weights(ensemble, ensemble_history)

        self.logger.info("Daily dynamic weight adjustment complete.")

    def _adjust_ensemble_weights(self, ensemble, history):
        base_model_scores = {}

        for model_name in ensemble.models.keys():
            if model_name in history.columns:
                predictions = history[model_name]
                actuals = history["actual"]
                score = self._calculate_performance_score(predictions, actuals)
                base_model_scores[model_name] = score

        if not base_model_scores:
            return

        current_weights = ensemble.ensemble_weights.copy()
        adjusted_weights = {}

        max_abs_score = max(abs(s) for s in base_model_scores.values())
        if max_abs_score == 0:
            return  # Avoid division by zero

        normalized_scores = {k: v / max_abs_score for k, v in base_model_scores.items()}

        for model_name, score in normalized_scores.items():
            adjustment_factor = score * self.max_adjustment
            current_weight = current_weights.get(model_name, 1.0 / len(current_weights))
            adjusted_weights[model_name] = max(
                0.01, current_weight * (1 + adjustment_factor)
            )  # Ensure weight doesn't go to 0

        total_adjusted_weight = sum(adjusted_weights.values())
        final_weights = {
            k: v / total_adjusted_weight for k, v in adjusted_weights.items()
        }

        self.logger.info(
            f"Updated weights for {ensemble.ensemble_name} from {current_weights} to {final_weights}"
        )
        ensemble.ensemble_weights = final_weights

    def _calculate_performance_score(self, predictions, actuals):
        score = 0
        for pred, actual in zip(predictions, actuals):
            if pd.isna(pred):
                continue

            if pred == actual and pred != "HOLD":
                score += self.correct_reward
            elif pred == "HOLD" and actual != "HOLD":
                score += self.missed_opportunity_penalty
            elif pred != actual:
                score += self.incorrect_penalty
        return score
