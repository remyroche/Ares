"""
Probability Calculation Framework

This module provides base probability calculation functions for different model types
and market scenarios. It serves as the foundation for generating the 4 required
probability outputs for the Enhanced Prediction Service.
"""

import numpy as np
import pandas as pd
import logging
from typing import Union
from sklearn.metrics import accuracy_score

logger = logging.getLogger(__name__)


class BaseProbabilityCalculator:

    @handle_errors(
        exceptions=(Exception,),
        default_return=False,
        context="baseprobabilitycalculator initialization",
    )
    async def initialize(self) -> bool:
        """Initialize BaseProbabilityCalculator."""
        try:
            self.logger.info(f"🚀 Initializing {class_name}...")
            self.is_initialized = True
            self.logger.info(f"✅ {class_name} initialized successfully")
            return True
        except Exception as e:
            self.logger.exception(f"❌ Error initializing {class_name}: {e}")
            return False
                """Base class for probability calculations across different model types."""

            # Get model predictions and probabilities
            y_pred_proba = model.predict_proba(X_test)
            confidence = self.calculate_confidence_from_proba(y_pred_proba)

            # Calculate market volatility
            if 'close' in market_data.columns:
                returns = market_data['close'].pct_change().dropna()
                volatility = returns.rolling(window=volatility_window).std().mean()
            else:
                volatility = 0.02  # Default volatility

            # Adjust probability based on volatility and target ratios
            volatility_factor = max(0.1, 1 - volatility * 10)
            target_ratio = profit_target / stop_loss

            # Base probability from model confidence
            base_prob = confidence * volatility_factor

            # Adjust for target ratio (higher ratio = lower probability)
            ratio_factor = min(1.0, 2.0 / target_ratio)

            final_prob = base_prob * ratio_factor
            return self.validate_probability(final_prob, "triple_barrier")

        except Exception as e:
            # Get predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)

            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)

            # Calculate confidence
            confidence = self.calculate_confidence_from_proba(y_pred_proba)

            # Combine accuracy and confidence
            direction_prob = (accuracy + confidence) / 2
            return self.validate_probability(direction_prob, "direction")

        except Exception as e:
            # Get prediction probabilities
            y_pred_proba = model.predict_proba(X_test)
            confidence = self.calculate_confidence_from_proba(y_pred_proba)

            # Calculate market volatility for magnitude context
            if 'close' in market_data.columns:
                returns = market_data['close'].pct_change().dropna()
                avg_magnitude = returns.abs().mean()
                volatility = returns.std()
            else:
                avg_magnitude = 0.01  # Default 1%
                volatility = 0.02

            # Adjust probability based on confidence and market conditions
            magnitude_prob = confidence * (1 - volatility * 5) * threshold_factor
            return self.validate_probability(magnitude_prob, "magnitude")

        except Exception as e:
            # Get prediction probabilities
            y_pred_proba = model.predict_proba(X_test)
            confidence = self.calculate_confidence_from_proba(y_pred_proba)

            # Calculate market risk metrics
            if 'close' in market_data.columns:
                returns = market_data['close'].pct_change().dropna()
                adverse_prob = (returns.abs() > adverse_threshold).mean()
                volatility = returns.std()
            else:
                adverse_prob = 0.3  # Default 30% chance of adverse movement
                volatility = 0.02

            # Calculate avoidance probability
            base_avoidance = 1 - adverse_prob
            volatility_adjustment = max(0.1, 1 - volatility * 10)

            avoidance_prob = base_avoidance * volatility_adjustment * confidence
            return self.validate_probability(avoidance_prob, "barrier_avoidance")

        except Exception as e:
            # Get predictions
            y_pred = model.predict(X_test)

            # Calculate prediction confidence based on variance
                y_pred_proba = model.predict_proba(X_test)
                confidence = self.calculate_confidence_from_proba(y_pred_proba)
            else:
# Use prediction magnitude as confidence proxy
                pred_magnitude = np.abs(y_pred)
                confidence = np.mean(np.minimum(pred_magnitude / 0.02, 1.0))

            # Calculate market volatility
            if 'close' in market_data.columns:
                returns = market_data['close'].pct_change().dropna()
                volatility = returns.std()
            else:
                volatility = 0.02

            # Adjust for volatility and target ratio
            volatility_factor = max(0.1, 1 - volatility * 10)
            target_ratio = profit_target / stop_loss
            ratio_factor = min(1.0, 2.0 / target_ratio)

            final_prob = confidence * volatility_factor * ratio_factor
            return self.validate_probability(final_prob, "triple_barrier")

        except Exception as e:
            # Get predictions
            y_pred = model.predict(X_test)

            # Calculate direction accuracy
            direction_correct = np.sign(y_pred) == np.sign(y_test)
            direction_accuracy = np.mean(direction_correct)

            # Calculate prediction confidence
            pred_magnitude = np.abs(y_pred)
            confidence = np.mean(np.minimum(pred_magnitude / 0.02, 1.0))

            # Combine accuracy and confidence
            direction_prob = (direction_accuracy + confidence) / 2
            return self.validate_probability(direction_prob, "direction")

        except Exception as e:
            # Get predictions
            y_pred = model.predict(X_test)
            predicted_magnitude = np.abs(y_pred)

            # Calculate actual magnitude from market data
            if 'close' in market_data.columns:
                actual_magnitude = np.abs(market_data['close'].pct_change().dropna())
                avg_actual_magnitude = actual_magnitude.mean()
            else:
                avg_actual_magnitude = 0.01  # Default 1%

            # Calculate magnitude accuracy
            magnitude_accuracy = np.mean(
                predicted_magnitude >= avg_actual_magnitude * threshold_factor
            )

            # Adjust for prediction confidence
            confidence = np.mean(np.minimum(predicted_magnitude / 0.02, 1.0))

            magnitude_prob = (magnitude_accuracy + confidence) / 2
            return self.validate_probability(magnitude_prob, "magnitude")

        except Exception as e:
            # Get predictions
            y_pred = model.predict(X_test)

            # Calculate prediction confidence
            pred_magnitude = np.abs(y_pred)
            confidence = np.mean(np.minimum(pred_magnitude / 0.02, 1.0))

            # Calculate market risk
            if 'close' in market_data.columns:
                returns = market_data['close'].pct_change().dropna()
                adverse_prob = (returns.abs() > adverse_threshold).mean()
                volatility = returns.std()
            else:
                adverse_prob = 0.3
                volatility = 0.02

            # Calculate avoidance probability
            base_avoidance = 1 - adverse_prob
            volatility_adjustment = max(0.1, 1 - volatility * 10)

            avoidance_prob = base_avoidance * volatility_adjustment * confidence
            return self.validate_probability(avoidance_prob, "barrier_avoidance")

        except Exception as e:
    elif model_type.lower() in ['regression', 'regressor', 'reg']:
                return RegressionProbabilityCalculator()
    else:
# Default to classification
        logger.warning(f"Unknown model type '{model_type}', defaulting to classification")
        return ClassificationProbabilityCalculator()