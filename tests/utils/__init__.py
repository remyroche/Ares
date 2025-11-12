"""
Module d'utilitaires pour les tests du projet ARES

Ce module contient des helpers et utilitaires pour standardiser
les assertions et améliorer la fiabilité des tests.
"""

from .assertions import (
    AssertionHelpers,
    assert_success_response,
    assert_error_response,
    assert_float_equals,
    assert_price_equals,
    assert_dict_structure,
    assert_execution_time,
    assert_order_status,
    assert_performance_metrics,
    assert_timestamp_format,
    assert_list_structure,
    assert_exchange_status,
    assert_signal_status,
    assert_dataframe_structure,
    assert_percentage_equals
)

__all__ = [
    'AssertionHelpers',
    'assert_success_response',
    'assert_error_response',
    'assert_float_equals',
    'assert_price_equals',
    'assert_dict_structure',
    'assert_execution_time',
    'assert_order_status',
    'assert_performance_metrics',
    'assert_timestamp_format',
    'assert_list_structure',
    'assert_exchange_status',
    'assert_signal_status',
    'assert_dataframe_structure',
    'assert_percentage_equals'
]