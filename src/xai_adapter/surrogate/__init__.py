"""Surrogate XAI methods for rules-vs-weights comparisons."""

from .base import CustomSurrogate, SurrogateMethod, make_surrogate
from .decision_tree import DecisionTreeSurrogateMethod
from .generator import (
    GeneratedSurrogateMethods,
    extract_tree_rules,
    generate_decision_tree_table,
    generate_logistic_regression_table,
    generate_surrogate_tables,
    rule_list_to_tree_structure,
)
from .logistic_regression import LogisticRegressionSurrogateMethod
from .rule_based import RuleListSurrogateMethod, RuleSetSurrogateMethod

__all__ = [
    "SurrogateMethod",
    "CustomSurrogate",
    "make_surrogate",
    "DecisionTreeSurrogateMethod",
    "LogisticRegressionSurrogateMethod",
    "RuleListSurrogateMethod",
    "RuleSetSurrogateMethod",
    "GeneratedSurrogateMethods",
    "extract_tree_rules",
    "generate_decision_tree_table",
    "generate_logistic_regression_table",
    "generate_surrogate_tables",
    "rule_list_to_tree_structure",
]
