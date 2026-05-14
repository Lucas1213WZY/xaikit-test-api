"""Rule-based surrogate methods for CoXAM-style explanation tables."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from ..base import ArrayLike
from .decision_tree import DecisionTreeSurrogateMethod


class _RuleBasedSurrogateMethod(DecisionTreeSurrogateMethod):
    """Shared base for rule-list and rule-set surrogates.

    Subclasses implement ``_order_rules`` to control the ordering of extracted
    root-to-leaf paths before they are encoded into the right-chained tree
    structure.  All traversal, prediction, and attribution logic is inherited
    from ``DecisionTreeSurrogateMethod`` unchanged.
    """

    def _order_rules(self, rules: List) -> List:
        raise NotImplementedError

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        *,
        app_id: Optional[str] = None,
        model_name: Optional[str] = None,
        depth: Optional[int] = None,
        random_state: Optional[int] = None,
        class_labels: Optional[List[Any]] = None,
        feature_names: Optional[List[str]] = None,
        **kwargs,
    ):
        from sklearn.metrics import accuracy_score
        from sklearn.tree import DecisionTreeClassifier

        from .generator import (
            _feature_names,
            _metadata_from_data,
            extract_tree_rules,
            rule_list_to_tree_structure,
        )

        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y)

        self.app_id = app_id or self.app_id or "custom_dataset"
        self.model_name = model_name or self.model_name or "external_model"
        self.depth = depth or self.depth
        self.random_state = self.random_state if random_state is None else random_state
        if class_labels is not None:
            self.class_labels = class_labels
        if feature_names is not None:
            self.feature_names = feature_names

        feat_names = _feature_names(X_arr.shape[1], self.feature_names)
        self.feature_names = feat_names

        clf = DecisionTreeClassifier(
            max_depth=self.depth,
            random_state=self.random_state,
            **{**self.tree_kwargs, **kwargs},
        )
        clf.fit(X_arr, y_arr)

        if self.class_labels is None:
            self.class_labels = clf.classes_.tolist()

        n_classes = len(clf.classes_)
        raw_rules = extract_tree_rules(clf.tree_, n_classes)
        self._rules = self._order_rules(raw_rules)

        counts = np.bincount(y_arr.astype(int), minlength=n_classes)
        default_class_idx = int(np.argmax(counts))

        self.tree_structure = rule_list_to_tree_structure(
            [(conds, cls) for conds, cls, _ in self._rules],
            default_class_idx,
            n_classes,
        )
        self.nodes_by_id = {node["node"]: node for node in self.tree_structure}
        self.fidelity = float(accuracy_score(y_arr, clf.predict(X_arr)))
        self.metadata_df = _metadata_from_data(
            X_arr, app_id=self.app_id, feature_names=feat_names
        )
        self.is_fitted = True
        return self

    def to_explanation_table(self):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not produce a CoXAM explanation table. "
            "Use get_rules() to inspect the extracted rules."
        )

    def get_rules(self) -> List[Dict[str, Any]]:
        """Return extracted rules as human-readable dicts."""
        self._require_fitted()
        return [
            {
                "rule_index": i,
                "conditions": (
                    " AND ".join(
                        f"{self.feature_names[f]} {'<=' if leq else '>'} {thresh:.4g}"
                        for f, thresh, leq in conditions
                    )
                    if conditions else "TRUE"
                ),
                "predicted_class": (
                    self.class_labels[class_idx] if self.class_labels else class_idx
                ),
                "confidence": confidence,
                "n_conditions": len(conditions),
            }
            for i, (conditions, class_idx, confidence) in enumerate(self._rules)
        ]


class RuleListSurrogateMethod(_RuleBasedSurrogateMethod):
    """Ordered rule-list surrogate (DFS left-to-right, first-match semantics)."""

    method_name = "rule_list"

    def _order_rules(self, rules: List) -> List:
        return rules


class RuleSetSurrogateMethod(_RuleBasedSurrogateMethod):
    """Unordered rule-set surrogate (confidence-sorted, highest-purity rule fires first)."""

    method_name = "rule_set"

    def _order_rules(self, rules: List) -> List:
        return sorted(rules, key=lambda r: (-r[2], len(r[0])))
