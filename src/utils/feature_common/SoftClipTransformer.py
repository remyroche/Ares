"""
Soft Clipping Transformer

Applies soft clipping using tanh to gently squash extreme values while
preserving the linear range in the middle. This is particularly useful
for preventing extreme outliers from dominating models while maintaining
gradient flow.
"""

from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd
from typing import Union


class SoftClipTransformer(BaseEstimator, TransformerMixin):
    """
    Soft clipping transformer using tanh function.

    Applies a soft clipping transformation that:
    - Preserves linearity in the range roughly [-3, 3] for stretch=2.0
    - Gently squashes extreme values (Z > 4) without hard cutoffs
    - Maintains differentiability for gradient-based optimization

    The transformation is: tanh(X / stretch)

    Parameters
    ----------
    stretch : float, default=2.0
        Controls the linear range width. Higher values create a wider linear zone.
        - stretch=2.0: linear range approximately [-3, +3], soft squashing beyond ±4
        - stretch=1.0: tighter squashing, linear range approximately [-1.5, +1.5]
        - stretch=3.0: wider linear range approximately [-4.5, +4.5]

    Examples
    --------
    >>> from SoftClipTransformer import SoftClipTransformer
    >>> import numpy as np
    >>>
    >>> # Create transformer
    >>> transformer = SoftClipTransformer(stretch=2.0)
    >>>
    >>> # Example data with extreme outliers
    >>> X = np.array([[-10, 0, 10], [5, 3, -5], [100, 2, -100]]).T
    >>>
    >>> # Transform
    >>> X_transformed = transformer.fit_transform(X)
    >>>
    >>> # Values within [-3, 3] are nearly linear
    >>> # Extreme values (±100) are squashed to approximately ±1
    """

    def __init__(self, stretch: float = 2.0):
        """
        Initialize soft clip transformer.

        Parameters
        ----------
        stretch : float, default=2.0
            Stretch parameter controlling the linear range width.
            Higher values preserve linearity over a wider range.
        """
        self.stretch = stretch

    def fit(self, X, y=None):
        """
        Fit the transformer (no-op, transformer is stateless).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        self : object
            Returns self.
        """
        return self

    def transform(self, X, y=None):
        """
        Apply soft clipping transformation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.
        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        X_transformed : ndarray or DataFrame of shape (n_samples, n_features)
            Soft-clipped data. Returns same type as input (ndarray or DataFrame).
        """
        # Handle pandas DataFrames
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(
                np.tanh(X.values / self.stretch),
                index=X.index,
                columns=X.columns
            )

        # Handle pandas Series
        elif isinstance(X, pd.Series):
            return pd.Series(
                np.tanh(X.values / self.stretch),
                index=X.index,
                name=X.name
            )

        # Handle numpy arrays
        else:
            X = np.asarray(X)
            return np.tanh(X / self.stretch)

    def inverse_transform(self, X):
        """
        Apply inverse soft clipping transformation.

        Note: This is the inverse tanh (arctanh), which can produce infinite
        values for inputs at ±1. Use with caution.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Soft-clipped data to inverse transform.

        Returns
        -------
        X_original : ndarray or DataFrame of shape (n_samples, n_features)
            Approximately original data (before soft clipping).
        """
        # Handle pandas DataFrames
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(
                np.arctanh(X.values) * self.stretch,
                index=X.index,
                columns=X.columns
            )

        # Handle pandas Series
        elif isinstance(X, pd.Series):
            return pd.Series(
                np.arctanh(X.values) * self.stretch,
                index=X.index,
                name=X.name
            )

        # Handle numpy arrays
        else:
            X = np.asarray(X)
            return np.arctanh(X) * self.stretch

    def get_params(self, deep=True):
        """
        Get parameters for this estimator.

        Parameters
        ----------
        deep : bool, default=True
            If True, will return the parameters for this estimator.

        Returns
        -------
        params : dict
            Parameter names mapped to their values.
        """
        return {"stretch": self.stretch}

    def set_params(self, **params):
        """
        Set the parameters of this estimator.

        Parameters
        ----------
        **params : dict
            Estimator parameters.

        Returns
        -------
        self : object
            Estimator instance.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self


__all__ = ['SoftClipTransformer']
