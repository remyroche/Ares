# src/analyst/base_regime_classifier.py

# src/analyst/base_regime_classifier.py
from abc import ABC, abstractmethod
import pandas as pd

class BaseRegimeClassifier(ABC):
    """
    Abstract base class for regime classifiers.
    Defines the common interface for training, predicting, and model persistence.
    """

    @abstractmethod
    def train(self, data: pd.DataFrame, **kwargs):
        """
        Train the regime classification model.

        Args:
            data (pd.DataFrame): DataFrame containing the training data (e.g., price, returns).
            **kwargs: Additional parameters for training.
        """
        pass

    @abstractmethod
    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Predict the regime for the given data.

        Args:
            data (pd.DataFrame): DataFrame for which to predict regimes.

        Returns:
            pd.DataFrame: The input DataFrame with an added 'regime' column.
        """
        pass

    @abstractmethod
    def save_model(self, filepath: str):
        """
        Save the trained model to a file.

        Args:
            filepath (str): The path to save the model file.
        """
        pass

    @abstractmethod
    def load_model(self, filepath: str):
        """
        Load a trained model from a file.

        Args:
            filepath (str): The path to the model file.
        """
        pass
