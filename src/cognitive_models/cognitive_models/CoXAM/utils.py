import json
import numpy as np
import pandas as pd
from collections import OrderedDict


class LogisticRegressionInterpreter:
    def __init__(self, explanation_df, metadata_df, app_id, model_name, variant="dense"):
        # 1) pick rows
        row = explanation_df[(explanation_df['appId'] == app_id) & (explanation_df['variant'] == variant)]

        if row.empty:
            raise ValueError(f"No logistic regression explanation found for appId: {app_id} and variant: {variant}")
        self.explanation_row = row.iloc[0]

        row = metadata_df[metadata_df['appId'] == app_id]
        if row.empty:
            raise ValueError(f"No metadata found for appId: {app_id}")
        self.metadata_row = row.iloc[0]

        self.app_id = app_id
        self.model = model_name
        self.fidelity = float(self.explanation_row['fidelity'])
        self._intercept_norm = float(self.explanation_row['intercept'])

        # 2) collect normalized-space coefficients in buckets by base index
        coef_keys = [
            k for k in self.explanation_row.index
            if k.startswith("coef_") and pd.notna(self.explanation_row[k])
        ]
        # coef_a{idx} (continuous) or coef_a{idx}={cat} (categorical)
        buckets = {}
        for k in coef_keys:
            val = float(self.explanation_row[k])
            name = k.replace("coef_", "")
            if "=" in name:
                base, cat = name.split("=")
                idx = int(base[1:])
                buckets.setdefault(idx, []).append(("cat", int(cat), val, name))
            else:
                idx = int(name[1:])
                buckets.setdefault(idx, []).append(("cont", None, val, name))

        # 3) collapse to RAW space (unnormalized inputs)
        icpt = float(self._intercept_norm)
        raw_coef_map = {}

        for idx, items in buckets.items():
            # classify bucket
            cats = [it for it in items if it[0] == "cat"]
            conts = [it for it in items if it[0] == "cont"]

            if len(conts) == 1 and len(cats) == 0:
                # Continuous feature (single normalized coef)
                c_norm = conts[0][2]

                vmin_key = f"v{idx}_min"
                vmax_key = f"v{idx}_max"
                vmin = self.metadata_row.get(vmin_key, None)
                vmax = self.metadata_row.get(vmax_key, None)

                if vmin is None or vmax is None or not np.isfinite(vmin) or not np.isfinite(vmax) or vmax == vmin:
                    # Fallback: can't unnormalize safely → keep as-is
                    raw_coef_map[f"a{idx}"] = c_norm
                else:
                    scale = (vmax - vmin)
                    c_raw = c_norm / scale
                    icpt -= (vmin * c_norm / scale)
                    raw_coef_map[f"a{idx}"] = c_raw

            elif len(cats) == 2 and all(c[1] in (0,1) for c in cats):
                # Binary categorical: collapse to single 0/1 feature
                c0 = next((v for (_t, cat, v, _n) in cats if cat == 0), None)
                c1 = next((v for (_t, cat, v, _n) in cats if cat == 1), None)
                if c0 is not None and c1 is not None and np.isfinite(c0) and np.isfinite(c1):
                    icpt += c0
                    raw_coef_map[f"a{idx}"] = (c1 - c0)
                else:
                    # Fallback: keep separate dummies
                    for (_t, cat, v, _n) in cats:
                        raw_coef_map[f"a{idx}={cat}"] = v

            else:
                # Multi-category or unusual bucket → keep as dummy variables
                for (_t, cat, v, _n) in items:
                    if cat is None:
                        # a rare case: both cont & cats present; just keep the normalized cont as-is
                        raw_coef_map[f"a{idx}"] = v
                    else:
                        raw_coef_map[f"a{idx}={cat}"] = v

        # 4) store in ordered form (a0, a1, a2, a2=0, a2=1, …)
        def sort_key(k):
            if "=" in k:
                base, cat = k.split("=")
                return (int(base[1:]), 1, int(cat), k)
            return (int(k[1:]), 0, -1, k)

        ordered_keys = sorted(raw_coef_map.keys(), key=sort_key)
        self.intercept = float(icpt)*100          # RAW-space intercept
        self.coefficients = OrderedDict((k, raw_coef_map[k]*100) for k in ordered_keys)

    def _format_feature(self, feature_key):
        if '=' in feature_key:
            base, cat_index = feature_key.split('=')
            v_index = base.replace('a', 'v')
            cat_col = f"{v_index}_{cat_index}"
            feat_name = self.metadata_row.get(base, base)
            cat_label = self.metadata_row.get(cat_col, f"Category {cat_index}")
            return f"{feat_name} = {cat_label}"
        else:
            return self.metadata_row.get(feature_key, feature_key)

    def __repr__(self, as_name=False):
        lines = [
            f"Logistic Regression (appId={self.app_id}, model={self.model})",
            f"Fidelity: {self.fidelity:.4f}",
            f"Intercept (raw): {self.intercept:.6g}",
            "Coefficients (raw inputs):"
        ]
        for key, val in self.coefficients.items():
            name = self._format_feature(key) if as_name else key
            lines.append(f"  {name:40} → {val:.6g}")
        return "\n".join(lines)

    def apply_to_instance(self, raw_input):
        """
        raw_input: numpy array of *raw* feature values ordered as a0, a1, a2, ...
                   For categoricals, raw_input[i] is the integer category index.
        """
        z = float(self.intercept)
        for key, coef in self.coefficients.items():
            if '=' in key:
                base, cat_idx = key.split('=')
                col_idx = int(base[1:])
                val = 1.0 if int(raw_input[col_idx]) == int(cat_idx) else 0.0
            else:
                col_idx = int(key[1:])
                val = raw_input[col_idx]
            z += coef * val
        return z

class DecisionTreeInterpreter:
    def __init__(self, explanation_df, metadata_df, app_id, model_name, depth=3):
        row = explanation_df[(explanation_df['appId'] == app_id) & (explanation_df['depth'] == depth)]
        if row.empty:
            raise ValueError(f"No decision tree explanation found for appId: {app_id}")
        self.explanation_row = row.iloc[0]

        row = metadata_df[metadata_df['appId'] == app_id]
        if row.empty:
            raise ValueError(f"No metadata found for appId: {app_id}")
        self.metadata_row = row.iloc[0]

        self.app_id = app_id
        self.model = model_name
        self.fidelity = self.explanation_row['fidelity']
        self.tree_structure = json.loads(self.explanation_row['tree_structure'])

        # ✅ Load and store class labels
        if "class_labels" in self.explanation_row:
            try:
                self.class_labels = json.loads(self.explanation_row["class_labels"])
            except Exception:
                self.class_labels = None
        else:
            self.class_labels = None

    def _format_feature(self, feature_key):
        if '=' in feature_key:
            base, cat_index = feature_key.split('=')
            v_index = base.replace('a', 'v')
            cat_col = f"{v_index}_{cat_index}"
            feat_name = self.metadata_row.get(base, base)
            cat_label = self.metadata_row.get(cat_col, f"Category {cat_index}")
            return f"{feat_name} = {cat_label}"
        else:
            return self.metadata_row.get(feature_key, feature_key)

    def print_tree(self, as_name=False):
        lines = [
            f"Decision Tree (appId={self.app_id}, model={self.model})",
            f"Fidelity: {self.fidelity:.4f}",
        ]
        self._print_node(0, 0, as_name, lines)
        return "\n".join(lines)

    def _print_node(self, node_id, depth, as_name, lines):
        node = next(n for n in self.tree_structure if n["node"] == node_id)
        prefix = "  " * depth
        if node["is_leaf"]:
            class_id = int(np.argmax(node["value"]))
            class_label = (
                self.class_labels[class_id]
                if self.class_labels and class_id < len(self.class_labels)
                else f"class {class_id}"
            )
            probs = np.round(node["value"], 4)
            lines.append(f"{prefix}-> Predict {class_label} (probs: {probs})")
        else:
            feature = self._format_feature(node['feature']) if as_name else node['feature']
            lines.append(f"{prefix}if {feature} <= {node['threshold']}:")
            self._print_node(node["left"], depth + 1, as_name, lines)
            lines.append(f"{prefix}else:")
            self._print_node(node["right"], depth + 1, as_name, lines)

    def apply_to_instance(self, raw_input):
        """
        raw_input: numpy array of feature values (ordered a0, a1, a2, ...)
        Returns: predicted class label or class distribution
        """
        node = next(n for n in self.tree_structure if n["node"] == 0)
        while not node["is_leaf"]:
            feature_key = node["feature"]
            if '=' in feature_key:
                base, cat_idx = feature_key.split('=')
                col_idx = int(base[1:])
                val = 1.0 if int(raw_input[col_idx]) == int(cat_idx) else 0.0
            else:
                col_idx = int(feature_key[1:])
                val = raw_input[col_idx]

            if val <= node["threshold"]:
                node = next(n for n in self.tree_structure if n["node"] == node["left"])
            else:
                node = next(n for n in self.tree_structure if n["node"] == node["right"])

        # return full probs and optionally the class label
        return {
            "probs": node["value"],
            "class_index": int(np.argmax(node["value"])),
            "class_label": self.class_labels[int(np.argmax(node["value"]))] if self.class_labels else None,
            "nid": node['node']
        }
    
    
    def simplify_tree(self, agg: str = "average"):
        """
        Collapse any internal node whose left and right are leaves with the same
        argmax class. Aggregates child probs by:
          - 'average' (default): (p_left + p_right) / 2
          - 'sum': p_left + p_right (then normalized)
          - 'left': use left child's probs
          - 'right': use right child's probs
        Modifies self.tree_structure in-place.
        """
        # index for quick lookup
        id2node = {n["node"]: n for n in self.tree_structure}

        # track nodes that become unreachable after merges
        reachable = set()

        def normalize(p):
            p = np.asarray(p, dtype=float)
            s = p.sum()
            return (p / s).tolist() if s > 0 else (np.ones_like(p) / len(p)).tolist()

        def aggregate(p_left, p_right, how):
            if how == "left":
                return p_left
            if how == "right":
                return p_right
            if how == "sum":
                return normalize((np.asarray(p_left) + np.asarray(p_right)).tolist())
            # default average
            return ((np.asarray(p_left) + np.asarray(p_right)) / 2.0).tolist()

        def recur(node_id):
            node = id2node[node_id]
            reachable.add(node_id)

            if node.get("is_leaf", False):
                # Return (is_leaf, class_idx)
                probs = np.asarray(node["value"], dtype=float)
                return True, int(np.argmax(probs))

            # simplify children first
            left_id, right_id = node["left"], node["right"]
            left_is_leaf, left_c = recur(left_id)
            right_is_leaf, right_c = recur(right_id)

            # if both leaves and same class -> collapse
            if left_is_leaf and right_is_leaf and (left_c == right_c):
                p_left = id2node[left_id]["value"]
                p_right = id2node[right_id]["value"]
                merged = aggregate(p_left, p_right, agg)

                # mutate current node into a leaf
                node.clear()
                node.update({
                    "node": node_id,
                    "is_leaf": True,
                    "value": merged
                })
                return True, left_c

            # otherwise keep as internal
            node["is_leaf"] = False
            return False, None

        # assume root has node id 0
        recur(0)

        # rebuild tree_structure to include only reachable nodes
        self.tree_structure = [id2node[i] for i in sorted(reachable)]


class AIDatasetLoader:
    def __init__(self, feature_values_df, metadata_df, AI_predictions_df):
        self.feature_values_df = feature_values_df
        self.metadata_df = metadata_df
        self.AI_predictions_df = AI_predictions_df

    def scale_value(self, value, min_val, max_val):
        if value < min_val:
            return 0
        elif value > max_val:
            return 1
        else:
            return (value - min_val) / (max_val - min_val)

    def scale_feature_values(self, instance_ids):
        scaled_features = []
        for instance_id in instance_ids:
            feature_row = self.feature_values_df[self.feature_values_df['instanceId'] == instance_id].iloc[0]
            app_id = feature_row['appId']
            app_metadata = self.metadata_df[self.metadata_df['appId'] == app_id]

            if app_metadata.empty:
                raise ValueError(f"No metadata found for appId: {app_id}")

            scaled_row = []
            i = 0
            while True:
                val_col = f'v{i}'
                min_col = f'v{i}_min'
                max_col = f'v{i}_max'

                if val_col not in feature_row.index or pd.isna(feature_row[val_col]):
                    break

                min_val = app_metadata[min_col].values[0] if min_col in app_metadata.columns else None
                max_val = app_metadata[max_col].values[0] if max_col in app_metadata.columns else None

                if pd.isna(min_val) and pd.isna(max_val):
                    scaled_row.append(feature_row[val_col])
                else:
                    value = feature_row[val_col]
                    scaled_row.append(self.scale_value(value, min_val, max_val))

                i += 1

            scaled_features.append(scaled_row)
        return scaled_features

    def _values_until_nan(self, feature_row):
        """Return [v0, v1, ..., vK] stopping at the first missing value."""
        vals = []
        i = 0
        while True:
            col = f"v{i}"
            if col not in feature_row.index:
                break
            val = feature_row[col]
            if pd.isna(val):
                break
            vals.append(val)
            i += 1
        return vals

    def load_instances(self, selected_ids, normalize=True):
        if selected_ids is None or selected_ids == []:
            raise ValueError("selected_ids must be provided and cannot be empty.")

        if normalize:
            scaled_features = self.scale_feature_values(selected_ids)
        else:
            scaled_features = []
            for instance_id in selected_ids:
                feature_row = self.feature_values_df[self.feature_values_df['instanceId'] == instance_id].iloc[0]
                scaled_row = self._values_until_nan(feature_row)
                scaled_features.append(scaled_row)

        AI_predictions = []
        for instance_id in selected_ids:
            pred_row = self.AI_predictions_df[self.AI_predictions_df['instanceId'] == instance_id]
            AI_predictions.append(pred_row.iloc[0]['pred'] if not pred_row.empty else None)

        return scaled_features, AI_predictions

    def filter_loader(self, condition):
        filtered_feature_values_df = self.feature_values_df[condition(self.feature_values_df)]
        filtered_metadata_df = self.metadata_df[condition(self.metadata_df)]
        filtered_predictions_df = self.AI_predictions_df[condition(self.AI_predictions_df)]

        return AIDatasetLoader(
            feature_values_df=filtered_feature_values_df,
            metadata_df=filtered_metadata_df,
            AI_predictions_df=filtered_predictions_df
        )

    def _iter_feature_indices(self, app_metadata_row: pd.Series) -> int:
        """Yield i for which v{i}_min/v{i}_max columns might exist (based on metadata columns)."""
        i = 0
        while True:
            min_col = f"v{i}_min"
            max_col = f"v{i}_max"
            val_col = f"v{i}"
            # stop when neither metadata min/max nor any value column appears
            if (min_col not in app_metadata_row.index and
                max_col not in app_metadata_row.index and
                val_col not in self.feature_values_df.columns):
                break
            yield i
            i += 1

    def _parse_categories_field(self, raw):
        """Accept list, JSON string, or python-literal string for categories."""
        if raw is None or (isinstance(raw, float) and np.isnan(raw)):
            return None
        if isinstance(raw, (list, tuple, np.ndarray)):
            return list(raw)
        if isinstance(raw, str):
            raw = raw.strip()
            try:
                # JSON first
                return list(json.loads(raw))
            except Exception:
                pass
            try:
                # python literal (e.g., "[0,1,2]" or "{...}")
                val = ast.literal_eval(raw)
                if isinstance(val, (list, tuple, set)):
                    return list(val)
            except Exception:
                pass
        return None

    def get_bounds_for_app(self, app_id, *, normalized: bool = False,
                           fallback_to_empirical: bool = True) -> dict:
        """
        Returns {'a{i}': (lo, hi)} for the app.
        - If normalized=True and metadata min/max exist -> (0.0, 1.0) for those features.
        - Else use metadata v{i}_min/v{i}_max when available.
        - If missing and fallback_to_empirical=True -> use empirical min/max from feature_values_df for that app.
        """
        md = self.metadata_df[self.metadata_df['appId'] == app_id]
        if md.empty:
            raise ValueError(f"No metadata found for appId: {app_id}")
        md_row = md.iloc[0]

        bounds = {}
        app_rows = self.feature_values_df[self.feature_values_df['appId'] == app_id]

        for i in self._iter_feature_indices(md_row):
            key = f"a{i}"
            val_col = f"v{i}"
            min_col = f"v{i}_min"
            max_col = f"v{i}_max"

            min_val = md_row[min_col] if min_col in md_row.index else np.nan
            max_val = md_row[max_col] if max_col in md_row.index else np.nan

            if normalized and pd.notna(min_val) and pd.notna(max_val):
                bounds[key] = (0.0, 1.0)
                continue

            if pd.notna(min_val) and pd.notna(max_val):
                bounds[key] = (float(min_val), float(max_val))
            elif fallback_to_empirical and val_col in app_rows.columns:
                col = app_rows[val_col].dropna()
                if not col.empty and np.issubdtype(col.dtype, np.number):
                    bounds[key] = (float(col.min()), float(col.max()))
                else:
                    # no numeric info; leave undefined for categorical (handled elsewhere)
                    pass
            # else: leave absent (categoricals / truly unknown)

        return bounds

    def get_categories_for_app(self, app_id, *, max_cardinality: int = 20) -> dict:
        """
        Returns {'a{i}': [int categories]} for the app.
        Priority:
          1) metadata columns like v{i}_categories / v{i}_levels / v{i}_cats if present
          2) infer from feature_values_df for the app:
             - treat integer-like or object columns with <= max_cardinality unique values as categorical
        """
        md = self.metadata_df[self.metadata_df['appId'] == app_id]
        if md.empty:
            raise ValueError(f"No metadata found for appId: {app_id}")
        md_row = md.iloc[0]
        app_rows = self.feature_values_df[self.feature_values_df['appId'] == app_id]

        categories = {}

        for i in self._iter_feature_indices(md_row):
            key = f"a{i}"
            # 1) metadata-driven
            meta_keys = [f"v{i}_categories", f"v{i}_levels", f"v{i}_cats", f"v{i}_domain"]
            cat_vals = None
            for mk in meta_keys:
                if mk in md_row.index:
                    cat_vals = self._parse_categories_field(md_row[mk])
                    if cat_vals:
                        break
            if cat_vals:
                categories[key] = sorted({int(v) for v in cat_vals if pd.notna(v)})
                continue

            # 2) data-driven inference
            vcol = f"v{i}"
            if vcol in app_rows.columns:
                col = app_rows[vcol].dropna()
                # integer-like or object with small cardinality
                uniq = pd.unique(col)
                if len(uniq) > 0 and len(uniq) <= max_cardinality:
                    # Must be all integer-like to be used as categorical ids
                    try:
                        cats = sorted({int(v) for v in uniq})
                        # Heuristic: if too many levels but still under limit and not all unique-per-row
                        if len(cats) <= max_cardinality:
                            categories[key] = cats
                    except Exception:
                        # non-integer categories; skip (your CF code expects ints)
                        pass

        return categories
    
    def get_feature_names(self, app_id) -> dict:
        md = self.metadata_df[self.metadata_df['appId'] == app_id]
        if md.empty:
            raise ValueError(f"No metadata found for appId: {app_id}")
        md_row = md.iloc[0]

        feature_names = {}
        for i in self._iter_feature_indices(md_row):
            key = f"a{i}"
            if key in md_row.index and pd.notna(md_row[key]):
                feature_names[key] = str(md_row[key])
            else:
                feature_names[key] = key

        return feature_names

def filter_by_app_and_model(ai_dataset_loader, app_id, model_name):
    # Apply filters based on expMethod and modelName
    def ai_condition(df):
        condition = pd.Series([True] * len(df), index=df.index)
        if 'appId' in df.columns:
            condition &= (df['appId'] == app_id)
        if 'modelName' in df.columns:
            condition &= (df['modelName'] == model_name)
        return condition


    return ai_dataset_loader.filter_loader(ai_condition)
