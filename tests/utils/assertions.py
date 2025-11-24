"""
Utilitaires d'assertion standardisés pour les tests du projet ARES

Ce module fournit des fonctions d'assertion standardisées pour garantir
la cohérence et la fiabilité des tests unitaires.
"""

try:
    import pytest  # type: ignore[import]
    PYTEST_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    pytest = None  # type: ignore[assignment]
    PYTEST_AVAILABLE = False

import numpy as np
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
import pandas as pd


class AssertionHelpers:
    """Classe contenant les helpers d'assertion standardisés."""
    
    # Constantes de tolérance pour les comparaisons numériques
    DEFAULT_FLOAT_TOLERANCE = 1e-6
    DEFAULT_PERCENTAGE_TOLERANCE = 0.01  # 1%
    DEFAULT_PRICE_TOLERANCE = 1e-4  # 0.01% pour les prix
    DEFAULT_TIME_TOLERANCE = 1.0  # 1 seconde
    
    @staticmethod
    def assert_success_response(
        response: Dict[str, Any],
        message: Optional[str] = None
    ) -> None:
        """
        Vérifie qu'une réponse API indique un succès.
        
        Args:
            response: Dictionnaire de réponse
            message: Message personnalisé pour l'erreur
        """
        assert response is not None, message or "La réponse ne doit pas être None"
        assert isinstance(response, dict), message or "La réponse doit être un dictionnaire"
        assert response.get('success') is True, message or f"La réponse devrait indiquer un succès: {response}"
    
    @staticmethod
    def assert_error_response(
        response: Dict[str, Any],
        expected_error_substring: Optional[str] = None,
        message: Optional[str] = None
    ) -> None:
        """
        Vérifie qu'une réponse API indique une erreur.
        
        Args:
            response: Dictionnaire de réponse
            expected_error_substring: Sous-chaîne attendue dans le message d'erreur
            message: Message personnalisé pour l'erreur
        """
        assert response is not None, message or "La réponse ne doit pas être None"
        assert isinstance(response, dict), message or "La réponse doit être un dictionnaire"
        assert response.get('success') is False, message or f"La réponse devrait indiquer une erreur: {response}"
        
        if expected_error_substring:
            error_msg = response.get('error', '').lower()
            assert expected_error_substring.lower() in error_msg, \
                message or f"L'erreur devrait contenir '{expected_error_substring}': {error_msg}"
    
    @staticmethod
    def assert_float_equals(
        actual: float,
        expected: float,
        tolerance: Optional[float] = None,
        message: Optional[str] = None
    ) -> None:
        """
        Vérifie l'égalité de deux nombres flottants avec tolérance.
        
        Args:
            actual: Valeur actuelle
            expected: Valeur attendue
            tolerance: Tolérance absolue (utilise DEFAULT_FLOAT_TOLERANCE si None)
            message: Message personnalisé pour l'erreur
        """
        if tolerance is None:
            tolerance = AssertionHelpers.DEFAULT_FLOAT_TOLERANCE
            
        diff = abs(actual - expected)
        assert diff <= tolerance, \
            message or f"Les valeurs ne correspondent pas: {actual} != {expected} (diff: {diff}, tol: {tolerance})"
    
    @staticmethod
    def assert_percentage_equals(
        actual: float,
        expected: float,
        tolerance: Optional[float] = None,
        message: Optional[str] = None
    ) -> None:
        """
        Vérifie l'égalité de deux pourcentages avec tolérance.
        
        Args:
            actual: Valeur actuelle (en décimal, ex: 0.05 pour 5%)
            expected: Valeur attendue (en décimal)
            tolerance: Tolérance (utilise DEFAULT_PERCENTAGE_TOLERANCE si None)
            message: Message personnalisé pour l'erreur
        """
        if tolerance is None:
            tolerance = AssertionHelpers.DEFAULT_PERCENTAGE_TOLERANCE
            
        diff = abs(actual - expected)
        assert diff <= tolerance, \
            message or f"Les pourcentages ne correspondent pas: {actual:.4f} != {expected:.4f} (diff: {diff:.4f}, tol: {tolerance:.4f})"
    
    @staticmethod
    def assert_price_equals(
        actual: float,
        expected: float,
        tolerance: Optional[float] = None,
        message: Optional[str] = None
    ) -> None:
        """
        Vérifie l'égalité de deux prix avec tolérance appropriée.
        
        Args:
            actual: Prix actuel
            expected: Prix attendu
            tolerance: Tolérance (utilise DEFAULT_PRICE_TOLERANCE si None)
            message: Message personnalisé pour l'erreur
        """
        if tolerance is None:
            tolerance = AssertionHelpers.DEFAULT_PRICE_TOLERANCE
            
        # Pour les prix, utiliser une tolérance relative
        relative_diff = abs(actual - expected) / max(abs(expected), abs(actual))
        assert relative_diff <= tolerance, \
            message or f"Les prix ne correspondent pas: {actual} != {expected} (diff rel: {relative_diff:.6f}, tol: {tolerance:.6f})"
    
    @staticmethod
    def assert_dict_structure(
        data: Dict[str, Any],
        required_keys: List[str],
        optional_keys: Optional[List[str]] = None,
        message: Optional[str] = None
    ) -> None:
        """
        Vérifie la structure d'un dictionnaire.
        
        Args:
            data: Dictionnaire à vérifier
            required_keys: Liste des clés requises
            optional_keys: Liste des clés optionnelles autorisées
            message: Message personnalisé pour l'erreur
        """
        assert isinstance(data, dict), message or "Les données doivent être un dictionnaire"
        
        missing_keys = [key for key in required_keys if key not in data]
        assert len(missing_keys) == 0, \
            message or f"Clés requises manquantes: {missing_keys} dans {list(data.keys())}"
        
        if optional_keys is not None:
            allowed_keys = set(required_keys + optional_keys)
            unexpected_keys = [key for key in data if key not in allowed_keys]
            assert len(unexpected_keys) == 0, \
                message or f"Clés inattendues trouvées: {unexpected_keys}"
    
    @staticmethod
    def assert_list_structure(
        data: List[Any],
        min_length: Optional[int] = None,
        max_length: Optional[int] = None,
        item_type: Optional[type] = None,
        message: Optional[str] = None
    ) -> None:
        """
        Vérifie la structure d'une liste.
        
        Args:
            data: Liste à vérifier
            min_length: Longueur minimale
            max_length: Longueur maximale
            item_type: Type attendu des éléments
            message: Message personnalisé pour l'erreur
        """
        assert isinstance(data, list), message or "Les données doivent être une liste"
        
        if min_length is not None:
            assert len(data) >= min_length, \
                message or f"La liste devrait avoir au moins {min_length} éléments (actuel: {len(data)})"
        
        if max_length is not None:
            assert len(data) <= max_length, \
                message or f"La liste devrait avoir au plus {max_length} éléments (actuel: {len(data)})"
        
        if item_type is not None:
            for i, item in enumerate(data):
                assert isinstance(item, item_type), \
                    message or f"L'élément {i} devrait être de type {item_type.__name__} (actuel: {type(item).__name__})"
    
    @staticmethod
    def assert_timestamp_format(
        timestamp: Union[str, datetime],
        format_type: str = "iso",
        message: Optional[str] = None
    ) -> None:
        """
        Vérifie le format d'un timestamp.
        
        Args:
            timestamp: Timestamp à vérifier
            format_type: Type de format ('iso', 'unix', 'datetime')
            message: Message personnalisé pour l'erreur
        """
        if format_type == "iso":
            if isinstance(timestamp, str):
                try:
                    datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                except ValueError:
                    assert False, message or f"Le timestamp ISO n'est pas valide: {timestamp}"
            else:
                assert isinstance(timestamp, datetime), \
                    message or f"Le timestamp devrait être une chaîne ISO ou datetime: {type(timestamp)}"
        
        elif format_type == "unix":
            assert isinstance(timestamp, (int, float)), \
                message or f"Le timestamp UNIX devrait être un nombre: {type(timestamp)}"
            assert isinstance(timestamp, (int, float)), \
                message or f"Le timestamp UNIX devrait être un nombre: {type(timestamp)}"
            assert float(timestamp) > 0, \
                message or f"Le timestamp UNIX devrait être positif: {timestamp}"
        
        elif format_type == "datetime":
            assert isinstance(timestamp, datetime), \
                message or f"Le timestamp devrait être un datetime: {type(timestamp)}"
    
    @staticmethod
    def assert_execution_time(
        execution_time: float,
        max_time: float,
        message: Optional[str] = None
    ) -> None:
        """
        Vérifie qu'un temps d'exécution est dans les limites attendues.
        
        Args:
            execution_time: Temps d'exécution en secondes
            max_time: Temps maximum autorisé
            message: Message personnalisé pour l'erreur
        """
        assert isinstance(execution_time, (int, float)), \
            message or f"Le temps d'exécution devrait être un nombre: {type(execution_time)}"
        assert execution_time >= 0, \
            message or f"Le temps d'exécution devrait être positif: {execution_time}"
        assert execution_time <= max_time, \
            message or f"Le temps d'exécution ({execution_time}s) dépasse la limite ({max_time}s)"
    
    @staticmethod
    def assert_order_status(
        actual_status: str,
        expected_status: str,
        message: Optional[str] = None
    ) -> None:
        """
        Vérifie le statut d'un ordre de manière standardisée.
        
        Args:
            actual_status: Statut actuel
            expected_status: Statut attendu
            message: Message personnalisé pour l'erreur
        """
        valid_statuses = ['PENDING', 'SUBMITTED', 'PARTIALLY_FILLED', 'FILLED', 'CANCELLED', 'REJECTED', 'EXPIRED']
        
        assert actual_status.upper() in valid_statuses, \
            message or f"Statut d'ordre invalide: {actual_status}. Valeurs valides: {valid_statuses}"
        
        assert actual_status.upper() == expected_status.upper(), \
            message or f"Le statut ne correspond pas: {actual_status} != {expected_status}"
    
    @staticmethod
    def assert_exchange_status(
        actual_status: str,
        expected_status: str,
        message: Optional[str] = None
    ) -> None:
        """
        Vérifie le statut d'un exchange de manière standardisée.
        
        Args:
            actual_status: Statut actuel
            expected_status: Statut attendu
            message: Message personnalisé pour l'erreur
        """
        valid_statuses = ['ACTIVE', 'INACTIVE', 'DISABLED', 'MAINTENANCE', 'ERROR']
        
        assert actual_status.upper() in valid_statuses, \
            message or f"Statut d'exchange invalide: {actual_status}. Valeurs valides: {valid_statuses}"
        
        assert actual_status.upper() == expected_status.upper(), \
            message or f"Le statut ne correspond pas: {actual_status} != {expected_status}"
    
    @staticmethod
    def assert_signal_status(
        actual_status: str,
        expected_status: str,
        message: Optional[str] = None
    ) -> None:
        """
        Vérifie le statut d'un signal de trading de manière standardisée.
        
        Args:
            actual_status: Statut actuel
            expected_status: Statut attendu
            message: Message personnalisé pour l'erreur
        """
        valid_statuses = ['RECEIVED', 'PROCESSED', 'FAILED', 'CANCELLED', 'TIMEOUT']
        
        assert actual_status.upper() in valid_statuses, \
            message or f"Statut de signal invalide: {actual_status}. Valeurs valides: {valid_statuses}"
        
        assert actual_status.upper() == expected_status.upper(), \
            message or f"Le statut ne correspond pas: {actual_status} != {expected_status}"
    
    @staticmethod
    def assert_performance_metrics(
        metrics: Dict[str, Any],
        required_metrics: Optional[List[str]] = None,
        message: Optional[str] = None
    ) -> None:
        """
        Vérifie la structure des métriques de performance.
        
        Args:
            metrics: Dictionnaire de métriques
            required_metrics: Liste des métriques requises
            message: Message personnalisé pour l'erreur
        """
        if required_metrics is None:
            required_metrics = [
                'total_return', 'sharpe_ratio', 'max_drawdown', 
                'volatility', 'win_rate', 'total_trades'
            ]
        
        AssertionHelpers.assert_dict_structure(
            metrics, 
            required_metrics, 
            message=message or f"Les métriques de performance doivent contenir: {required_metrics}"
        )
        
        # Vérifier que les métriques numériques sont valides
        for metric in ['total_return', 'sharpe_ratio', 'max_drawdown', 'volatility', 'win_rate']:
            if metric in metrics:
                value = metrics[metric]
                assert isinstance(value, (int, float)), \
                    message or f"La métrique {metric} devrait être numérique: {type(value)}"
                assert not (isinstance(value, float) and np.isnan(value)), \
                    message or f"La métrique {metric} ne devrait pas être NaN"
    
    @staticmethod
    def assert_dataframe_structure(
        df: pd.DataFrame,
        expected_columns: Optional[List[str]] = None,
        min_rows: Optional[int] = None,
        max_rows: Optional[int] = None,
        message: Optional[str] = None
    ) -> None:
        """
        Vérifie la structure d'un DataFrame pandas.
        
        Args:
            df: DataFrame à vérifier
            expected_columns: Colonnes attendues
            min_rows: Nombre minimum de lignes
            max_rows: Nombre maximum de lignes
            message: Message personnalisé pour l'erreur
        """
        assert isinstance(df, pd.DataFrame), message or "Les données doivent être un DataFrame pandas"
        
        if expected_columns is not None:
            missing_cols = [col for col in expected_columns if col not in df.columns]
            assert len(missing_cols) == 0, \
                message or f"Colonnes manquantes: {missing_cols} dans {list(df.columns)}"
        
        if min_rows is not None:
            assert len(df) >= min_rows, \
                message or f"Le DataFrame devrait avoir au moins {min_rows} lignes (actuel: {len(df)})"
        
        if max_rows is not None:
            assert len(df) <= max_rows, \
                message or f"Le DataFrame devrait avoir au plus {max_rows} lignes (actuel: {len(df)})"


# Fonctions globales pour un accès facile
def assert_success_response(response: Dict[str, Any], message: Optional[str] = None) -> None:
    """Wrapper global pour AssertionHelpers.assert_success_response."""
    return AssertionHelpers.assert_success_response(response, message)


def assert_error_response(response: Dict[str, Any], expected_error_substring: Optional[str] = None, message: Optional[str] = None) -> None:
    """Wrapper global pour AssertionHelpers.assert_error_response."""
    return AssertionHelpers.assert_error_response(response, expected_error_substring, message)


def assert_float_equals(actual: float, expected: float, tolerance: Optional[float] = None, message: Optional[str] = None) -> None:
    """Wrapper global pour AssertionHelpers.assert_float_equals."""
    return AssertionHelpers.assert_float_equals(actual, expected, tolerance, message)


def assert_price_equals(actual: float, expected: float, tolerance: Optional[float] = None, message: Optional[str] = None) -> None:
    """Wrapper global pour AssertionHelpers.assert_price_equals."""
    return AssertionHelpers.assert_price_equals(actual, expected, tolerance, message)


def assert_dict_structure(data: Dict[str, Any], required_keys: List[str], optional_keys: Optional[List[str]] = None, message: Optional[str] = None) -> None:
    """Wrapper global pour AssertionHelpers.assert_dict_structure."""
    return AssertionHelpers.assert_dict_structure(data, required_keys, optional_keys, message)


def assert_execution_time(execution_time: float, max_time: float, message: Optional[str] = None) -> None:
    """Wrapper global pour AssertionHelpers.assert_execution_time."""
    return AssertionHelpers.assert_execution_time(execution_time, max_time, message)


def assert_order_status(actual_status: str, expected_status: str, message: Optional[str] = None) -> None:
    """Wrapper global pour AssertionHelpers.assert_order_status."""
    return AssertionHelpers.assert_order_status(actual_status, expected_status, message)


def assert_performance_metrics(metrics: Dict[str, Any], required_metrics: Optional[List[str]] = None, message: Optional[str] = None) -> None:
    """Wrapper global pour AssertionHelpers.assert_performance_metrics."""
    return AssertionHelpers.assert_performance_metrics(metrics, required_metrics, message)


def assert_timestamp_format(timestamp: Union[str, datetime], format_type: str = "iso", message: Optional[str] = None) -> None:
    """Wrapper global pour AssertionHelpers.assert_timestamp_format."""
    return AssertionHelpers.assert_timestamp_format(timestamp, format_type, message)


def assert_list_structure(data: List[Any], min_length: Optional[int] = None, max_length: Optional[int] = None, item_type: Optional[type] = None, message: Optional[str] = None) -> None:
    """Wrapper global pour AssertionHelpers.assert_list_structure."""
    return AssertionHelpers.assert_list_structure(data, min_length, max_length, item_type, message)


def assert_exchange_status(actual_status: str, expected_status: str, message: Optional[str] = None) -> None:
    """Wrapper global pour AssertionHelpers.assert_exchange_status."""
    return AssertionHelpers.assert_exchange_status(actual_status, expected_status, message)


def assert_signal_status(actual_status: str, expected_status: str, message: Optional[str] = None) -> None:
    """Wrapper global pour AssertionHelpers.assert_signal_status."""
    return AssertionHelpers.assert_signal_status(actual_status, expected_status, message)


def assert_dataframe_structure(df: pd.DataFrame, expected_columns: Optional[List[str]] = None, min_rows: Optional[int] = None, max_rows: Optional[int] = None, message: Optional[str] = None) -> None:
    """Wrapper global pour AssertionHelpers.assert_dataframe_structure."""
    return AssertionHelpers.assert_dataframe_structure(df, expected_columns, min_rows, max_rows, message)


def assert_percentage_equals(actual: float, expected: float, tolerance: Optional[float] = None, message: Optional[str] = None) -> None:
    """Wrapper global pour AssertionHelpers.assert_percentage_equals."""
    return AssertionHelpers.assert_percentage_equals(actual, expected, tolerance, message)


def assert_true(condition: bool, message: Optional[str] = None) -> None:
    """Vérifie qu'une condition est vraie."""
    assert bool(condition), message or "Expected condition to be True"


def assert_false(condition: bool, message: Optional[str] = None) -> None:
    """Vérifie qu'une condition est fausse."""
    assert not bool(condition), message or "Expected condition to be False"


def assert_equals(actual: Any, expected: Any, message: Optional[str] = None) -> None:
    """Vérifie l'égalité simple entre deux valeurs."""
    assert actual == expected, message or f"Expected {expected!r}, got {actual!r}"


def assert_not_equals(actual: Any, expected: Any, message: Optional[str] = None) -> None:
    """Vérifie que deux valeurs sont différentes."""
    assert actual != expected, message or f"Did not expect {expected!r}, but got the same value"


def assert_greater_than(actual: float, threshold: float, message: Optional[str] = None) -> None:
    """Vérifie que actual > threshold."""
    assert actual > threshold, message or f"Expected {actual} > {threshold}"


def assert_less_than(actual: float, threshold: float, message: Optional[str] = None) -> None:
    """Vérifie que actual < threshold."""
    assert actual < threshold, message or f"Expected {actual} < {threshold}"


def assert_greater_than_or_equal(actual: float, threshold: float, message: Optional[str] = None) -> None:
    """Vérifie que actual >= threshold."""
    assert actual >= threshold, message or f"Expected {actual} >= {threshold}"


def assert_less_than_or_equal(actual: float, threshold: float, message: Optional[str] = None) -> None:
    """Vérifie que actual <= threshold."""
    assert actual <= threshold, message or f"Expected {actual} <= {threshold}"


def assert_array_shape(arr: np.ndarray, expected_shape: Any, message: Optional[str] = None) -> None:
    """Vérifie la forme d'un tableau numpy."""
    assert isinstance(arr, np.ndarray), message or f"Expected numpy array, got {type(arr)}"
    assert tuple(arr.shape) == tuple(expected_shape), message or f"Expected shape {expected_shape}, got {arr.shape}"


def assert_array_not_empty(arr: np.ndarray, message: Optional[str] = None) -> None:
    """Vérifie qu'un tableau numpy n'est pas vide."""
    assert isinstance(arr, np.ndarray), message or f"Expected numpy array, got {type(arr)}"
    assert arr.size > 0, message or "Expected non-empty array"


def assert_array_no_nan(arr: np.ndarray, message: Optional[str] = None) -> None:
    """Vérifie qu'un tableau numpy ne contient pas de NaN."""
    assert isinstance(arr, np.ndarray), message or f"Expected numpy array, got {type(arr)}"
    assert not np.isnan(arr).any(), message or "Array contains NaN values"


def assert_array_no_inf(arr: np.ndarray, message: Optional[str] = None) -> None:
    """Vérifie qu'un tableau numpy ne contient pas d'infinis."""
    assert isinstance(arr, np.ndarray), message or f"Expected numpy array, got {type(arr)}"
    assert not np.isinf(arr).any(), message or "Array contains inf/-inf values"


def assert_dtype(arr: np.ndarray, expected_dtype: Any, message: Optional[str] = None) -> None:
    """Vérifie le dtype d'un tableau numpy."""
    assert isinstance(arr, np.ndarray), message or f"Expected numpy array, got {type(arr)}"
    assert arr.dtype == expected_dtype, message or f"Expected dtype {expected_dtype}, got {arr.dtype}"


def assert_in_range(value: float, min_value: float, max_value: float, message: Optional[str] = None) -> None:
    """Vérifie qu'une valeur est dans [min_value, max_value]."""
    assert min_value <= value <= max_value, message or f"Expected {value} in range [{min_value}, {max_value}]"


def assert_is_none(value: Any, message: Optional[str] = None) -> None:
    """Vérifie qu'une valeur est None."""
    assert value is None, message or f"Expected value to be None, got {value!r}"


def assert_is_not_none(value: Any, message: Optional[str] = None) -> None:
    """Vérifie qu'une valeur n'est pas None."""
    assert value is not None, message or "Expected value to be not None"


def assert_contains(container: Any, item: Any, message: Optional[str] = None) -> None:
    """Vérifie qu'un élément est contenu dans un conteneur."""
    assert item in container, message or f"Expected {item!r} to be in {container!r}"


def assert_not_contains(container: Any, item: Any, message: Optional[str] = None) -> None:
    """Vérifie qu'un élément n'est pas contenu dans un conteneur."""
    assert item not in container, message or f"Did not expect {item!r} to be in {container!r}"


def assert_in(container: Any, item: Any, message: Optional[str] = None) -> None:
    """Alias pratique pour assert_contains."""
    assert_contains(container, item, message)


def assert_not_in(container: Any, item: Any, message: Optional[str] = None) -> None:
    """Alias pratique pour assert_not_contains."""
    assert_not_contains(container, item, message)


def assert_is_instance(value: Any, expected_type: type, message: Optional[str] = None) -> None:
    """Vérifie qu'une valeur est instance d'un type donné."""
    assert isinstance(value, expected_type), message or f"Expected {value!r} to be instance of {expected_type.__name__}"


def assert_is_not_instance(value: Any, unexpected_type: type, message: Optional[str] = None) -> None:
    """Vérifie qu'une valeur n'est pas instance d'un type donné."""
    assert not isinstance(value, unexpected_type), message or f"Did not expect {value!r} to be instance of {unexpected_type.__name__}"


def assert_array_dtype(arr: np.ndarray, expected_dtype: Any, message: Optional[str] = None) -> None:
    """Vérifie le dtype d'un tableau numpy (alias pour assert_dtype)."""
    assert_dtype(arr, expected_dtype, message)


def assert_array_range(arr: np.ndarray, min_value: float, max_value: float, message: Optional[str] = None) -> None:
    """Vérifie que toutes les valeurs du tableau sont dans [min_value, max_value]."""
    assert isinstance(arr, np.ndarray), message or f"Expected numpy array, got {type(arr)}"
    within_bounds = (arr >= min_value) & (arr <= max_value)
    assert bool(within_bounds.all()), message or f"Array values not all in range [{min_value}, {max_value}]"


def assert_string_contains(text: str, substring: str, message: Optional[str] = None) -> None:
    """Vérifie qu'une sous-chaîne est présente dans une chaîne."""
    assert substring in text, message or f"Expected '{substring}' to be in '{text}'"


def assert_string_not_contains(text: str, substring: str, message: Optional[str] = None) -> None:
    """Vérifie qu'une sous-chaîne n'est pas présente dans une chaîne."""
    assert substring not in text, message or f"Did not expect '{substring}' to be in '{text}'"


def assert_file_exists(path: str, message: Optional[str] = None) -> None:
    """Vérifie qu'un fichier existe."""
    import os
    assert os.path.isfile(path), message or f"Expected file to exist: {path}"


def assert_directory_exists(path: str, message: Optional[str] = None) -> None:
    """Vérifie qu'un répertoire existe."""
    import os
    assert os.path.isdir(path), message or f"Expected directory to exist: {path}"


def assert_key_exists(mapping: Dict[Any, Any], key: Any, message: Optional[str] = None) -> None:
    """Vérifie qu'une clé existe dans un dictionnaire."""
    assert key in mapping, message or f"Expected key {key!r} to exist in mapping"


def assert_key_not_exists(mapping: Dict[Any, Any], key: Any, message: Optional[str] = None) -> None:
    """Vérifie qu'une clé n'existe pas dans un dictionnaire."""
    assert key not in mapping, message or f"Did not expect key {key!r} to exist in mapping"