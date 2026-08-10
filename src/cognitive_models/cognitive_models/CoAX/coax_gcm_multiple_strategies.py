import math
import numpy as np
import copy
import random
import itertools

class MemoryBase:
    """
    Each exemplar is stored as:
       {
         'label_probs': {label -> probability},
         'features': [...],
         'explanation': [... or None],
         'time_stored': float
       }

    The retrieve_exemplars(...) method returns a list of (ex, activation) pairs,
    where activation = -decay_param * ln(time_elapsed). Only exemplars with activation
    above the retrieval_threshold are returned.
    """
    def __init__(self, decay_param=0.5, retrieval_threshold=None):
        self._exemplars = []  # Made private to avoid direct external access.
        self.decay_param = decay_param
        self.retrieval_threshold = retrieval_threshold

    def store_exemplar(self, label_probs, features, explanation, time_stored):
        ex = {
            'label_probs': copy.deepcopy(label_probs),
            'features': copy.deepcopy(features),
            'explanation': copy.deepcopy(explanation),
            'time_stored': time_stored
        }
        self._exemplars.append(ex)

    def retrieve_exemplars(self, current_time):
        """
        Return a list of (ex, activation) pairs, where activation is computed as:
            activation = -decay_param * ln(time_elapsed)
        Only exemplars with activation >= retrieval_threshold (if set) are returned.
        """
        out = []
        for ex in self._exemplars:
            time_elapsed = max(1e-12, current_time - ex['time_stored'])
            activation = -self.decay_param * math.log(time_elapsed)
            # Apply the retrieval threshold if it is set.
            if self.retrieval_threshold is not None and activation < self.retrieval_threshold:
                continue
            out.append((ex, activation))
        return out


###############################################################################
# Utility Functions
 
def euclidean_distance(vec1, vec2):
    """
    Compute Euclidean distance between two vectors, ignoring None values.
    Clip element-wise differences to [-1, 1]. Normalize by sqrt(# valid dims).
    Return None if no valid dims.
    """
    arr1 = np.array([x if x is not None else np.nan for x in vec1])
    arr2 = np.array([x if x is not None else np.nan for x in vec2])

    valid = ~(np.isnan(arr1) | np.isnan(arr2))
    num_valid = np.sum(valid)
    if num_valid == 0:
        return None

    diffs = arr1[valid] - arr2[valid]
    clipped_diffs = np.clip(diffs, -1, 1)
    dist = np.linalg.norm(clipped_diffs)
    normalized_dist = dist / math.sqrt(num_valid)
    return normalized_dist


def compute_similarity(dist, activation, sensitivity=1.0, temperature=1.0):
    """
    GCM/EBRW-style similarity:
      sim = exp(-sensitivity * dist + activation),
    where activation = -(decay_param)*ln(time_elapsed).
    """
    return math.exp((-sensitivity * dist + activation)/temperature)

def normalize_label_strengths(label_strengths):
    """
    Convert raw strengths to a probability distribution.
    Fallback to uniform if sum is near zero.
    """
    total = sum(label_strengths.values())
    if total < 1e-12:
        if len(label_strengths) == 0:
            return {}
        n = len(label_strengths)
        return {lbl: 1.0/n for lbl in label_strengths}
    else:
        return {lbl: val / total for lbl, val in label_strengths.items()}

# def compute_decision_time(label_strengths, base_time=0, time_scale=2.0, epsilon=1e-1):
#     """
#     EBRW-inspired decision time:
#       T = base_time + time_scale / max(delta, epsilon)
#     where delta = (largest label strength) - (second largest).
#     If only one label is present, delta = that single strength.
#     """
#     if not label_strengths:
#         return base_time
#     sorted_vals = sorted(label_strengths.values(), reverse=True)
#     if len(sorted_vals) == 1:
#         delta = sorted_vals[0]
#     else:
#         delta = sorted_vals[0] - sorted_vals[1]
#     return base_time + time_scale / max(delta, epsilon)




class HumanModelBase:
    """
    Base class that handles placeholders for last inference distribution and data.
    Subclasses implement new_instance(), infer(), feedback().

    infer() and feedback() take a `trial_context` dict assembled by the caller from
    the dataset, the explanation table, and the generated trial row:
        'features'      -> feature values for the instance (from the dataset)
        'explanation'   -> explanation vector, or None when none is shown
        'ai_prediction' -> the AI prediction the trial carries
    """
    def __init__(self, time=None, decay_param=0.5):
        self.time = time
        self.decay_param = decay_param
        self.all_labels = set()

        self.last_inference_probs = None
        self.last_features = None
        self.last_explanation = None

        # Typical time costs (base overhead)
        self.constant_inference_time = 0.1
        self.constant_feedback_time = 1.0

    def new_instance(self):
        raise NotImplementedError

    def infer(self, trial_context):
        raise NotImplementedError

    def feedback(self, trial_context):
        raise NotImplementedError

    def _default_response(self):
        if not self.all_labels:
            return {0: 0.5, 1: 0.5}
        n = len(self.all_labels)
        return {lbl: 1.0/n for lbl in self.all_labels}

class SensitiveFeatures(HumanModelBase):
    def __init__(self, time=None, decay_param=0.5,
                 retrieval_threshold=-2.5,
                 sensitivity=10.0,
                 k=3,
                 **kwargs
                 ):
        super().__init__(time=time, decay_param=decay_param)
        
        self.memory = MemoryBase(decay_param=decay_param, retrieval_threshold=retrieval_threshold)
        self.sensitivity = sensitivity
        self.k = k

        self.constant_time = 5

    def select_focus_indices(self, features):
        """
        Automatically select focus indices based on a t-test metric.

        Fallback logic:
        - If there are no active exemplars, or
        - If there are not exactly two label groups, or
        - If either group has ≤ 1 exemplar,
        then select feature indices at random.
        """
        n_features = len(features)
        n_focus = max(1, int(self.k))

        def _random_focus_indices():
            indices = list(range(n_features))
            random.shuffle(indices)
            self.focus_indices = indices

        # Get the current time (needed for retrieval)
        current_time = self.time.get_time() if self.time else 0.0
        # Retrieve only active exemplars
        active_exemplars = [
            ex for (ex, activation) in self.memory.retrieve_exemplars(current_time=current_time)
        ]

        # Fallback 1: no active exemplars
        if not active_exemplars:
            return _random_focus_indices()

        # Group active exemplars by their most-probable label.
        groups = {}
        for exemplar in active_exemplars:
            if exemplar['label_probs']=={}:
                print(exemplar)
                print(active_exemplars)
            label = max(exemplar['label_probs'], key=exemplar['label_probs'].get)
            groups.setdefault(label, []).append(exemplar['features'])

        # Fallback 2: not exactly two groups
        if len(groups) != 2:
            return _random_focus_indices()

        # For binary classification, extract the two groups.
        group_keys = list(groups.keys())
        group1_features = np.array(groups[group_keys[0]])  # shape: (n1, n_features)
        group2_features = np.array(groups[group_keys[1]])  # shape: (n2, n_features)
        n1 = group1_features.shape[0]
        n2 = group2_features.shape[0]

        # Fallback 3: any group too small for a stable t-statistic
        if n1 <= 1 or n2 <= 1:
            return _random_focus_indices()

        # Compute means for each feature in each group.
        means1 = np.nanmean(group1_features, axis=0)
        means2 = np.nanmean(group2_features, axis=0)

        # Compute variances (sample variance, ddof=1), then apply a lower bound.
        vars1 = np.nanvar(group1_features, axis=0, ddof=1)
        vars2 = np.nanvar(group2_features, axis=0, ddof=1)

        # Set a lower bound for the variance to avoid zero errors.
        var_eps = 1e-8
        vars1 = np.maximum(vars1, var_eps)
        vars2 = np.maximum(vars2, var_eps)

        # Compute the standard error for each feature.
        se = np.sqrt(vars1 / n1 + vars2 / n2)
        # Extra safeguard (should rarely trigger with variance floor).
        se[se == 0] = 1e-12

        # Compute the absolute t-statistic.
        t_stats = np.abs(means1 - means2) / se

        # Rank indices by the t-statistic in descending order.
        sorted_indices = np.argsort(t_stats)[::-1]
        self.focus_indices = sorted_indices[:n_focus].tolist()

    def new_instance(self):
        # If we have a last inference not yet stored, store it as a "soft-labeled" exemplar.
        if self.last_inference_probs is not None and self.last_features is not None:
            current_time = self.time.get_time() if self.time else 0.0
            if self.last_inference_probs == {}:
                print(self.focus_indices)
                print("Warning: last_inference_probs is empty in SensitiveFeatures.new_instance()")
            # if self.save_new_exemplar:
            self.memory.store_exemplar(
                label_probs=self.last_inference_probs,
                features=self.last_features,
                explanation=self.last_explanation,
                time_stored=current_time
            )
            self.time.add_time(self.constant_time)
        self.last_inference_probs = None
        self.last_features = None
        self.last_explanation = None

    def infer(self, trial_context):
        start_time = self.time.get_time()

        features = trial_context.get('features')
        explanation = trial_context.get('explanation')

        self.select_focus_indices(features)
        focus_indices = self.focus_indices


        current_time = self.time.get_time() if self.time else 0.0
        stored = self.memory.retrieve_exemplars(
            current_time=current_time
        )

        if self.time:
            self.time.add_time(self.constant_inference_time)

        if not stored:
            probs = self._default_response()
            self.last_inference_probs = probs
            total_time = self.time.get_time() - start_time
            return probs, total_time
        
        # Create test vector masked to only include focus features
        test_vec = np.array(
            [features[i] if i in focus_indices else None for i in range(len(features))],
            dtype=float
        )
        self.last_features = features

        label_strengths = {}


        for (ex, activation) in stored:
            dist = euclidean_distance(test_vec, ex["features"])
            if dist is None:
                continue

            sim = compute_similarity(dist, activation, self.sensitivity, 1.0)

            # Accumulate weighted label probabilities
            for lbl, p_lbl in ex['label_probs'].items():
                label_strengths[lbl] = label_strengths.get(lbl, 0.0) + sim * p_lbl

        # Convert strengths into normalized probabilities
        probs = normalize_label_strengths(label_strengths)
        if probs == {}:
            print("Warning: probs is empty in SensitiveFeatures.infer()")
        self.last_inference_probs = probs
        
        return probs, self.time.get_time() - start_time

    def feedback(self, trial_context):
        features = trial_context.get('features')
        explanation = trial_context.get('explanation')
        ai_pred = trial_context.get('ai_prediction')

        label_probs = {ai_pred: 1.0}
        current_time = self.time.get_time() if self.time else 0.0

        # Store a "hard-labeled" exemplar.
        self.memory.store_exemplar(
            label_probs=label_probs,
            features=features,
            explanation=explanation,
            time_stored=current_time
        )

        self.last_inference_probs = None
        self.last_features = None
        self.last_explanation = None


        if self.time:
            self.time.add_time(self.constant_time)
        return self.constant_time

class SalientFeatures(HumanModelBase):
    """
    Zero out features except top-k by explanation magnitude.
    Uses MemoryBase.
    Includes a constant time cost for inference and feedback.
    """

    def __init__(
        self, time=None, decay_param=0.5,
        k=3,
        sensitivity=10.0,
        retrieval_threshold=-2.5,
        **kwargs
    ):
        super().__init__(time=time, decay_param=decay_param)
        self.memory = MemoryBase(decay_param=decay_param, retrieval_threshold=retrieval_threshold)
        self.k = k
        self.sensitivity = sensitivity

        # Simplified constant times
        self.constant_time = 5.0

    def new_instance(self):
        # If we have a last inference not yet stored, store it as a "soft-labeled" exemplar.
        if self.last_inference_probs is not None and self.last_features is not None:
            current_time = self.time.get_time() if self.time else 0.0
            top_k_features = self._mask_top_k_features(self.last_features, self.last_explanation)
            self.memory.store_exemplar(
                label_probs=self.last_inference_probs,
                    features=top_k_features,
                    explanation=self.last_explanation,
                    time_stored=current_time
                )
            if self.time:
                self.time.add_time(self.constant_time)

        self.last_inference_probs = None
        self.last_features = None
        self.last_explanation = None

    def infer(self, trial_context):
        start_time = self.time.get_time() if self.time else 0.0

        features = trial_context.get('features')
        explanation = trial_context.get('explanation')

        self.last_explanation = explanation

        # Mask top-k features (by explanation magnitude) if explanation is present
        if explanation is not None:
            top_k_indices = np.argsort(explanation)[-self.k:]
            masked_test_vec = [
                features[i] if i in top_k_indices else None
                for i in range(len(features))
            ]
        else:
            masked_test_vec = features

        self.last_features = masked_test_vec

        # Retrieve stored exemplars
        current_time = self.time.get_time() if self.time else 0.0
        stored_exemplars = self.memory.retrieve_exemplars(current_time=current_time)

        
        if self.time:
            self.time.add_time(self.constant_time)

        if not stored_exemplars:
            probs = self._default_response()
            self.last_inference_probs = probs
            total_time = (self.time.get_time() - start_time)
            return probs, total_time

        test_vec = np.array(masked_test_vec, dtype=float)
        label_strengths = {}
        valid_exemplar_found = False

        for (ex, activation) in stored_exemplars:
            # Mask exemplar features similarly (if it has an explanation)
            if ex.get('explanation') is not None:
                ex_top_k_indices = np.argsort(ex['explanation'])[-self.k:]
                masked_ex_features = [
                    ex['features'][i] if i in ex_top_k_indices else None
                    for i in range(len(ex['features']))
                ]
            else:
                masked_ex_features = ex['features']

            ex_vec = np.array(masked_ex_features, dtype=float)
            dist = euclidean_distance(test_vec, ex_vec)
            if dist is None:
                continue
            valid_exemplar_found = True

            sim = compute_similarity(dist, activation, self.sensitivity, 1.0)

            for lbl, p_lbl in ex['label_probs'].items():
                label_strengths[lbl] = label_strengths.get(lbl, 0.0) + sim * p_lbl

        if not valid_exemplar_found:
            probs = self._default_response()
            self.last_inference_probs = probs
            total_time = (self.time.get_time() - start_time)
            return probs, total_time

        probs = normalize_label_strengths(label_strengths)
        self.last_inference_probs = probs

        
        return probs, (self.time.get_time() - start_time)

    def feedback(self, trial_context):
        features = trial_context.get('features')
        explanation = trial_context.get('explanation')
        ai_pred = trial_context.get('ai_prediction')

        top_k_features = self._mask_top_k_features(features, explanation)
        label_probs = {ai_pred: 1.0}
        current_time = self.time.get_time() if self.time else 0.0

        self.memory.store_exemplar(
            label_probs=label_probs,
            features=top_k_features,
            explanation=explanation,
            time_stored=current_time
        )
        self.all_labels.add(ai_pred)

        self.last_inference_probs = None
        self.last_features = None
        self.last_explanation = None

        if self.time:
            self.time.add_time(self.constant_time)
        return self.constant_time

    def _mask_top_k_features(self, features, explanation):
        if explanation is None:
            return features
        top_k_indices = np.argsort(explanation)[-self.k:]
        return [
            features[i] if i in top_k_indices else None
            for i in range(len(features))
        ]

    

class ImportanceCategorization(HumanModelBase):
    """
    Ignores raw features, uses explanation only (MemoryBase).
    Activation-based similarity; simplified timing with a single constant time.
    """

    def __init__(
        self, time=None, decay_param=0.5,
        sensitivity=10.0,
        k=3,                    # number of explanation indices to attend to
        retrieval_threshold=-2.5,
        **kwargs
    ):
        super().__init__(time=time, decay_param=decay_param)
        self.memory = MemoryBase(decay_param=decay_param,
                                 retrieval_threshold=retrieval_threshold)
        self.sensitivity = sensitivity
        self.k = k

        # Single constant time cost used for both inference and feedback
        self.constant_time = 5.0

    def select_focus_indices(self, explanation):
        """
        Select focus indices using a t-test metric when possible.
        If there are not exactly two label groups or groups are too small,
        simply attend to ALL indices (no random selection).
        """
        n_dims = len(explanation)
        n_focus = max(1, int(self.k))

        # Default: use all features
        def _all_indices():
            return list(range(n_dims))

        current_time = self.time.get_time() if self.time else 0.0

        # Retrieve active exemplars with explanation
        retrieved = self.memory.retrieve_exemplars(current_time=current_time)
        active_exemplars = [
            ex for (ex, activation) in retrieved
            if ex.get('explanation') is not None
        ]

        # Fallback 1: no active exemplars → use all
        if not active_exemplars:
            return _all_indices()

        # Group by most-probable label
        groups = {}
        for exemplar in active_exemplars:
            label = max(exemplar['label_probs'], key=exemplar['label_probs'].get)
            groups.setdefault(label, []).append(exemplar['explanation'])

        # Fallback 2: not exactly two groups → use all
        if len(groups) != 2:
            return _all_indices()

        # Extract groups
        labels = list(groups.keys())
        group1 = np.array(groups[labels[0]])
        group2 = np.array(groups[labels[1]])

        n1 = group1.shape[0]
        n2 = group2.shape[0]

        # Fallback 3: not enough samples → use all
        if n1 <= 1 or n2 <= 1:
            return _all_indices()

        # Compute means
        means1 = np.mean(group1, axis=0)
        means2 = np.mean(group2, axis=0)

        # Compute variances (standard formula)
        vars1 = np.var(group1, axis=0, ddof=1)
        vars2 = np.var(group2, axis=0, ddof=1)

        # Compute SE
        se = np.sqrt(vars1/n1 + vars2/n2)
        se[se == 0] = 1e-12

        # t-statistics
        t_stats = np.abs(means1 - means2) / se

        # Sort by decreasing t
        sorted_indices = np.argsort(t_stats)[::-1]
        return sorted_indices[:n_focus].tolist()


    def new_instance(self):
        # Store the last inference as a soft-labeled exemplar.
        if self.last_inference_probs is not None and self.last_features is not None:
            current_time = self.time.get_time() if self.time else 0.0
            self.memory.store_exemplar(
                label_probs=self.last_inference_probs,
                features=self.last_features,        # raw features
                explanation=self.last_explanation,  # explanation vector
                time_stored=current_time
            )
            if self.time:
                self.time.add_time(self.constant_time)

        self.last_inference_probs = None
        self.last_features = None
        self.last_explanation = None

    def infer(self, trial_context):
        start_time = self.time.get_time() if self.time else 0.0

        features = trial_context.get('features')
        explanation = trial_context.get('explanation')

        self.last_features = features
        self.last_explanation = explanation

        if self.time:
            self.time.add_time(self.constant_time)

        # Fallback if no explanation
        if explanation is None:
            probs = self._default_response()
            self.last_inference_probs = probs
            return probs, (self.time.get_time() - start_time)
        
        # Select indices to attend to
        focus_indices = self.select_focus_indices(explanation)
        if not focus_indices:
            probs = self._default_response()
            self.last_inference_probs = probs
            return probs, (self.time.get_time() - start_time)

        # Retrieve stored exemplars
        current_time = self.time.get_time() if self.time else 0.0
        stored = self.memory.retrieve_exemplars(current_time=current_time)

        if not stored:
            probs = self._default_response()
            self.last_inference_probs = probs
            return probs, (self.time.get_time() - start_time)

        test_vec = np.array([explanation[i] for i in focus_indices], dtype=float)
        label_strengths = {}
        valid_exemplar_found = False

        for ex, activation in stored:
            if ex.get('explanation') is None:
                continue

            ex_vec = np.array([ex['explanation'][i] for i in focus_indices], dtype=float)
            dist = euclidean_distance(test_vec, ex_vec)
            if dist is None:
                continue
            valid_exemplar_found = True

            sim = compute_similarity(dist, activation, self.sensitivity, 1.0)

            for lbl, p_lbl in ex['label_probs'].items():
                label_strengths[lbl] = label_strengths.get(lbl, 0.0) + sim * p_lbl

        if not valid_exemplar_found or not label_strengths:
            probs = self._default_response()
            self.last_inference_probs = probs
            return probs, (self.time.get_time() - start_time)

        probs = normalize_label_strengths(label_strengths)
        self.last_inference_probs = probs


        return probs, (self.time.get_time() - start_time)

    def feedback(self, trial_context):
        features = trial_context.get('features')
        explanation = trial_context.get('explanation')
        ai_pred = trial_context.get('ai_prediction')

        label_probs = {ai_pred: 1.0}
        current_time = self.time.get_time() if self.time else 0.0

        self.memory.store_exemplar(
            label_probs=label_probs,
            features=features,
            explanation=explanation,
            time_stored=current_time
        )
        self.all_labels.add(ai_pred)

        self.last_inference_probs = None
        self.last_features = None
        self.last_explanation = None

        if self.time:
            self.time.add_time(self.constant_time)
        return self.constant_time

class AttributionSum(HumanModelBase):
    """
    Attribution-summing strategy using MemoryBase.

    Modes via `explanation_type`:
      - 'importance':
          Uses memory; feedback stores exemplars labeled by ai_prediction.
      - 'attribution':
          Ignores memory at inference; sums explanation directly and
          feedback labels features by explanation sign.
    """

    def __init__(
        self,
        time=None,
        decay_param=0.5,
        retrieval_threshold=-0.3,
        sensitivity=15.0,
        scaling_factor=1.0,
        k=2,
        explanation_type='importance',
        **kwargs
    ):
        super().__init__(time=time, decay_param=decay_param)
        self.memory = MemoryBase(decay_param=decay_param,
                                 retrieval_threshold=retrieval_threshold)

        self.sensitivity = sensitivity
        self.scaling_factor = scaling_factor
        self.k = k

        # Mode selector: 'importance' or 'attribution'
        self.explanation_type = explanation_type

        # Single constant time cost for inference and feedback
        self.constant_time = 5.0

        # NEW: store per-feature importance/attribution for last instance
        self.last_importances = None

    def new_instance(self):
        """
        If last_inference_probs + last_features exist:
        store top-k single-feature exemplars (soft-labeled).
        Each stored exemplar has all features = None except one dimension.

        Now:
        - If self.last_importances is available, use it to select top-k
          and store it in the exemplar's 'explanation' field.
        """
        if self.last_inference_probs is not None and self.last_features is not None:
            current_time = self.time.get_time()

            if self.last_importances is not None:
                # Use imputed/partial importance vector to pick top-k
                idx_val_imp = [
                    (i, self.last_features[i], self.last_importances[i])
                    for i in range(len(self.last_features))
                ]
                top_k = sorted(
                    idx_val_imp,
                    key=lambda x: abs(x[2]),
                    reverse=True
                )[:self.k]
            else:
                # Fallback: no importance info → store all features
                top_k = [
                    (i, self.last_features[i], None)
                    for i in range(len(self.last_features))
                ]

            for (i, val, imp_val) in top_k:
                feat_arr = [None] * len(self.last_features)
                feat_arr[i] = val

                # Store the full importance vector as 'explanation'
                explanation_vector = self.last_importances

                self.memory.store_exemplar(
                    label_probs=self.last_inference_probs,
                    features=feat_arr,
                    explanation=explanation_vector,
                    time_stored=current_time
                )

            self.time.add_time(self.constant_time)

        self.last_inference_probs = None
        self.last_features = None
        self.last_explanation = None
        self.last_importances = None   # reset

    def infer(self, trial_context):
        """
        Perform one inference and return (probs, total_time).

        - If explanation_type == 'attribution' and explanation is present:
            Skip memory; sum explanation (top-k) directly.
        - Else:
            Use memory and importance-based or imputed importance logic.

        Additionally:
        - Maintain self.last_importances as a per-feature vector of:
            * partial attribution when explanation is present
            * imputed importance when explanation is absent
        """
        start_time = self.time.get_time()

        features = trial_context.get('features')
        explanation = trial_context.get('explanation')

        self.last_features = features
        self.last_explanation = explanation
        self.last_importances = None

        if features is not None:
            # initialize vector for this instance
            self.last_importances = [0.0] * len(features)

        # Case 1: Pure attribution mode, explanation given:
        # sum top-k explanation values directly and pass through logistic.
        if self.explanation_type == 'attribution' and explanation is not None:
            top_k_indices = np.argsort(np.abs(explanation))[::-1][:self.k]
            total_attr = sum(explanation[i] for i in top_k_indices)

            # Store per-feature attribution = explanation (simple choice)
            if self.last_importances is not None:
                for i in range(len(explanation)):
                    self.last_importances[i] = explanation[i]

            prob_label_1 = 1.0 / (1.0 + math.exp(-self.scaling_factor * total_attr))
            prob_label_0 = 1.0 - prob_label_1
            probs = {0: prob_label_0, 1: prob_label_1}

            self.last_inference_probs = probs
            self.time.add_time(self.constant_time)
            total_infer_time = self.time.get_time() - start_time
            return probs, total_infer_time

        # Case 2: Use memory (importance explanations or imputation)
        total_attribution = 0.0
        current_time = self.time.get_time()

        # 2a) Explanation present, explanation_type == 'importance':
        #     use top-k |explanation| indices; derive partial attributions.
        if explanation is not None and self.explanation_type == 'importance':
            top_k_indices = np.argsort(np.abs(explanation))[::-1][:self.k]
            top_k = [(i, features[i], explanation[i]) for i in top_k_indices]

            for (i, feat_val, exp_val) in top_k:
                stored_exemplars = self.memory.retrieve_exemplars(
                    current_time=current_time
                )

                local_strengths = {}
                for (ex, activation) in stored_exemplars:
                    if i < len(ex['features']) and ex['features'][i] is not None:
                        dist = abs(feat_val - ex['features'][i])
                        sim = math.exp(-self.sensitivity * dist + activation)
                        for lbl, p_lbl in ex['label_probs'].items():
                            local_strengths[lbl] = local_strengths.get(lbl, 0.0) + sim * p_lbl

                if not local_strengths:
                    continue

                local_dist = normalize_label_strengths(local_strengths)
                raw_sign = sum(
                    (+1.0 if lbl == 1 else -1.0) * p_lbl
                    for lbl, p_lbl in local_dist.items()
                )

                if abs(raw_sign) < 0.01:
                    continue

                expected_sign = 1.0 if raw_sign > 0 else -1.0
                partial_attr = expected_sign * exp_val
                total_attribution += partial_attr

                # Store per-feature partial attribution
                if self.last_importances is not None:
                    self.last_importances[i] = partial_attr

        # 2b) No explanation: impute importance by similarity-weighted averaging over all features.
        elif explanation is None and features is not None:
            for i, feat_val in enumerate(features):
                stored_exemplars = self.memory.retrieve_exemplars(
                    current_time=current_time
                )

                sim_weighted_sum = 0.0
                total_sim = 0.0

                for ex, activation in stored_exemplars:
                    if i < len(ex['features']) and ex['features'][i] is not None:
                        ex_val = ex['features'][i]
                        dist = abs(feat_val - ex_val)
                        sim = math.exp(-self.sensitivity * dist + activation)

                        label_probs = ex['label_probs']
                        label = max(label_probs, key=label_probs.get)
                        label_sign = 1.0 if label == 1 else -1.0

                        explanation_vector = ex.get("explanation")
                        if explanation_vector is None or i >= len(explanation_vector):
                            continue

                        raw_exp_val = explanation_vector[i]

                        # Flip sign if label and explanation disagree
                        exp_val = raw_exp_val
                        if math.copysign(1, raw_exp_val) != label_sign:
                            exp_val *= -1

                        sim_weighted_sum += sim * exp_val
                        total_sim += sim

                if total_sim > 0:
                    imputed_importance = sim_weighted_sum / total_sim
                else:
                    imputed_importance = 0.0

                total_attribution += imputed_importance

                # Store per-feature imputed importance
                if self.last_importances is not None:
                    self.last_importances[i] = imputed_importance

        total_attribution = max(-1e3, min(1e3, total_attribution))

        # Logistic to get probabilities
        prob_label_1 = 1.0 / (1.0 + math.exp(-total_attribution * self.scaling_factor))
        prob_label_0 = 1.0 - prob_label_1
        probs = {0: prob_label_0, 1: prob_label_1}
        self.last_inference_probs = probs

        self.time.add_time(self.constant_time)
        total_infer_time = self.time.get_time() - start_time
        return probs, total_infer_time

    def feedback(self, trial_context):
        """
        Feedback updates memory with single-feature exemplars.

        - If explanation_type == 'importance':
            Use ai_prediction as label for all stored exemplars.
        - If explanation_type == 'attribution':
            Label each stored exemplar by sign of its explanation
            (+ => label=1, - => label=0).

        When explanation is provided → only top-k |explanation| dims stored.
        When explanation is None     → all features stored.
        """
        start_time = self.time.get_time()

        features = trial_context.get('features')
        explanation = trial_context.get('explanation')
        ai_pred = trial_context.get('ai_prediction')

        if self.explanation_type == 'importance':
            label_probs_global = {ai_pred: 1.0}
        else:
            label_probs_global = None  # per-feature in attribution mode

        current_time = self.time.get_time()

        if explanation is not None:
            # Use top-k |explanation| indices
            top_k_indices = np.argsort(np.abs(explanation))[::-1][:self.k]
            top_k = [(i, features[i], explanation[i]) for i in top_k_indices]
        else:
            # No explanation → use all features
            top_k = [(i, features[i], None) for i in range(len(features))]

        for (i, val, exp_val) in top_k:
            feat_arr = [None] * len(features)
            feat_arr[i] = val

            if self.explanation_type == 'attribution':
                if exp_val is not None and exp_val >= 0:
                    local_label_probs = {1: 1.0}
                else:
                    local_label_probs = {0: 1.0}
            else:
                local_label_probs = label_probs_global

            self.memory.store_exemplar(
                label_probs=local_label_probs,
                features=feat_arr,
                explanation=explanation,   # unchanged feedback logic
                time_stored=current_time
            )
            self.all_labels.update(local_label_probs.keys())

        self.last_inference_probs = None
        self.last_features = None
        self.last_explanation = None
        self.last_importances = None

        self.time.add_time(self.constant_time)
        total_time = self.time.get_time() - start_time
        return total_time

# class AttributionSum(HumanModelBase):
#     """
#     Attribution-summing strategy using MemoryBase.

#     Modes via `explanation_type`:
#       - 'importance':
#           Uses memory; feedback stores exemplars labeled by ai_prediction.
#       - 'attribution':
#           Ignores memory at inference; sums explanation directly and
#           feedback labels features by explanation sign.
#     """

#     def __init__(
#         self,
#         time=None,
#         decay_param=0.5,
#         retrieval_threshold=-0.3,
#         sensitivity=15.0,
#         scaling_factor=1.0,
#         k=2,
#         explanation_type='importance',
#         **kwargs
#     ):
#         super().__init__(time=time, decay_param=decay_param)
#         self.memory = MemoryBase(decay_param=decay_param,
#                                  retrieval_threshold=retrieval_threshold)

#         self.sensitivity = sensitivity
#         self.scaling_factor = scaling_factor
#         self.k = k

#         # Mode selector: 'importance' or 'attribution'
#         self.explanation_type = explanation_type

#         # Single constant time cost for inference and feedback
#         self.constant_time = 5.0

#     def new_instance(self):
#         """
#         If last_inference_probs + last_features exist:
#         store top-k single-feature exemplars (soft-labeled).
#         Each stored exemplar has all features = None except one dimension.
#         """
#         if self.last_inference_probs is not None and self.last_features is not None:
#             current_time = self.time.get_time()

#             if self.last_explanation is not None:
#                 # pick top-k by explanation magnitude
#                 idx_val_exp = [
#                     (i, self.last_features[i], self.last_explanation[i])
#                     for i in range(len(self.last_features))
#                 ]
#                 top_k = sorted(
#                     idx_val_exp,
#                     key=lambda x: abs(x[2]),
#                     reverse=True
#                 )[:self.k]
#             else:
#                 # no explanation => store all features
#                 top_k = [
#                     (i, self.last_features[i], None)
#                     for i in range(len(self.last_features))
#                 ]

#             for (i, val, exp_val) in top_k:
#                 feat_arr = [None] * len(self.last_features)
#                 feat_arr[i] = val

#                 self.memory.store_exemplar(
#                     label_probs=self.last_inference_probs,
#                     features=feat_arr,
#                     explanation=None,
#                     time_stored=current_time
#                 )

#             self.time.add_time(self.constant_time)

#         self.last_inference_probs = None
#         self.last_features = None
#         self.last_explanation = None

#     def infer(self, ui):
#         """
#         Perform one inference and return (probs, total_time).

#         - If explanation_type == 'attribution' and explanation is present:
#             Skip memory; sum explanation (top-k) directly.
#         - Else:
#             Use memory and importance-based or imputed importance logic.
#         """
#         start_time = self.time.get_time()

#         features = ui.get_value('features')
#         explanation = ui.get_value('explanation')

    
#         # if explanation is not None:
#         #     print("Explanation provided for AttributionSum inference.")
#         # else:
#         #     print("No explanation provided for AttributionSum inference.")

#         self.last_features = features
#         self.last_explanation = explanation

#         # Case 1: Pure attribution mode, explanation given:
#         # sum top-k explanation values directly and pass through logistic.
#         if self.explanation_type == 'attribution' and explanation is not None:
#             top_k_indices = np.argsort(np.abs(explanation))[::-1][:self.k]
#             total_attr = sum(explanation[i] for i in top_k_indices)

#             prob_label_1 = 1.0 / (1.0 + math.exp(-self.scaling_factor * total_attr))
#             prob_label_0 = 1.0 - prob_label_1
#             probs = {0: prob_label_0, 1: prob_label_1}

#             self.last_inference_probs = probs
#             self.time.add_time(self.constant_time)
#             total_infer_time = self.time.get_time() - start_time
#             return probs, total_infer_time

#         # Case 2: Use memory (importance explanations or imputation)
#         total_attribution = 0.0
#         current_time = self.time.get_time()

#         # 2a) Explanation present, explanation_type == 'importance':
#         #     use top-k |explanation| indices.
#         if explanation is not None and self.explanation_type == 'importance':
#             top_k_indices = np.argsort(np.abs(explanation))[::-1][:self.k]
#             top_k = [(i, features[i], explanation[i]) for i in top_k_indices]

#             for (i, feat_val, exp_val) in top_k:
#                 stored_exemplars = self.memory.retrieve_exemplars(
#                     current_time=current_time
#                 )

#                 local_strengths = {}
#                 for (ex, activation) in stored_exemplars:
#                     if i < len(ex['features']) and ex['features'][i] is not None:
#                         dist = abs(feat_val - ex['features'][i])
#                         sim = math.exp(-self.sensitivity * dist + activation)
#                         for lbl, p_lbl in ex['label_probs'].items():
#                             local_strengths[lbl] = local_strengths.get(lbl, 0.0) + sim * p_lbl

#                 if not local_strengths:
#                     continue

#                 local_dist = normalize_label_strengths(local_strengths)
#                 raw_sign = sum(
#                     (+1.0 if lbl == 1 else -1.0) * p_lbl
#                     for lbl, p_lbl in local_dist.items()
#                 )

#                 if abs(raw_sign) < 0.01:
#                     continue

#                 expected_sign = 1.0 if raw_sign > 0 else -1.0
#                 partial_attr = expected_sign * exp_val
#                 total_attribution += partial_attr

#         # 2b) No explanation: impute importance by similarity-weighted averaging over all features.
#         elif explanation is None and features is not None:
#             for i, feat_val in enumerate(features):
#                 stored_exemplars = self.memory.retrieve_exemplars(
#                     current_time=current_time
#                 )


#                 sim_weighted_sum = 0.0
#                 total_sim = 0.0

#                 for ex, activation in stored_exemplars:
#                     if i < len(ex['features']) and ex['features'][i] is not None:
#                         ex_val = ex['features'][i]
#                         dist = abs(feat_val - ex_val)
#                         sim = math.exp(-self.sensitivity * dist + activation)

#                         label_probs = ex['label_probs']
#                         label = max(label_probs, key=label_probs.get)
#                         label_sign = 1.0 if label == 1 else -1.0

#                         explanation_vector = ex.get("explanation")
#                         if explanation_vector is None or i >= len(explanation_vector):
#                             continue

#                         raw_exp_val = explanation_vector[i]

#                         # Flip sign if label and explanation disagree
#                         exp_val = raw_exp_val
#                         if math.copysign(1, raw_exp_val) != label_sign:
#                             exp_val *= -1

#                         sim_weighted_sum += sim * exp_val
#                         total_sim += sim
                
#                 if total_sim!=0:
#                     print("total_sim:", total_sim)
#                     import sys
#                     # sys.exit()

#                 if total_sim > 0:
#                     imputed_importance = sim_weighted_sum / total_sim
#                 else:
#                     imputed_importance = 0.0

#                 total_attribution += imputed_importance

#         # Logistic to get probabilities
#         prob_label_1 = 1.0 / (1.0 + math.exp(-total_attribution * self.scaling_factor))
#         prob_label_0 = 1.0 - prob_label_1
#         probs = {0: prob_label_0, 1: prob_label_1}
#         self.last_inference_probs = probs

#         self.time.add_time(self.constant_time)
#         total_infer_time = self.time.get_time() - start_time
#         return probs, total_infer_time

#     def feedback(self, ui):
#         """
#         Feedback updates memory with single-feature exemplars.

#         - If explanation_type == 'importance':
#             Use ai_prediction as label for all stored exemplars.
#         - If explanation_type == 'attribution':
#             Label each stored exemplar by sign of its explanation
#             (+ => label=1, - => label=0).

#         When explanation is provided → only top-k |explanation| dims stored.
#         When explanation is None     → all features stored.
#         """
#         start_time = self.time.get_time()

#         features = ui.get_value('features')
#         explanation = ui.get_value('explanation')
#         ai_pred = ui.get_value('ai_prediction')

#         if self.explanation_type == 'importance':
#             label_probs_global = {ai_pred: 1.0}
#         else:
#             label_probs_global = None  # per-feature in attribution mode

#         current_time = self.time.get_time()

#         if explanation is not None:
#             # Use top-k |explanation| indices
#             top_k_indices = np.argsort(np.abs(explanation))[::-1][:self.k]
#             top_k = [(i, features[i], explanation[i]) for i in top_k_indices]
#         else:
#             # No explanation → use all features
#             top_k = [(i, features[i], None) for i in range(len(features))]

#         for (i, val, exp_val) in top_k:
#             feat_arr = [None] * len(features)
#             feat_arr[i] = val

#             if self.explanation_type == 'attribution':
#                 if exp_val is not None and exp_val >= 0:
#                     local_label_probs = {1: 1.0}
#                 else:
#                     local_label_probs = {0: 1.0}
#             else:
#                 local_label_probs = label_probs_global

#             self.memory.store_exemplar(
#                 label_probs=local_label_probs,
#                 features=feat_arr,
#                 explanation=explanation,
#                 time_stored=current_time
#             )
#             self.all_labels.update(local_label_probs.keys())

#         self.last_inference_probs = None
#         self.last_features = None
#         self.last_explanation = None

#         self.time.add_time(self.constant_time)
#         total_time = self.time.get_time() - start_time
#         return total_time

# class AttributionSum(HumanModelBase):
#     """
#     Attribution-summing strategy using MemoryBase.

#     Modes via `explanation_type`:
#       - 'importance':
#           Uses memory; feedback stores exemplars labeled by ai_prediction.
#       - 'attribution':
#           Ignores memory at inference; sums explanation directly and
#           feedback labels features by explanation sign.
#     """

#     def __init__(
#         self,
#         time=None,
#         decay_param=0.5,
#         retrieval_threshold=-0.3,
#         sensitivity=15.0,
#         scaling_factor=1.0,
#         k=2,
#         explanation_type='importance',
#         **kwargs
#     ):
#         super().__init__(time=time, decay_param=decay_param)
#         self.memory = MemoryBase(decay_param=decay_param,
#                                  retrieval_threshold=retrieval_threshold)

#         self.sensitivity = sensitivity
#         self.scaling_factor = scaling_factor
#         self.k = k

#         # Mode selector: 'importance' or 'attribution'
#         self.explanation_type = explanation_type

#         # Single constant time cost for inference and feedback
#         self.constant_time = 5.0

#     def new_instance(self):
#         """
#         If last_inference_probs + last_features exist:
#         store top-k single-feature exemplars (soft-labeled).
#         Each stored exemplar has all features = None except one dimension.
#         """
#         if self.last_inference_probs is not None and self.last_features is not None:
#             current_time = self.time.get_time()

#             if self.last_explanation is not None:
#                 # pick top-k by explanation magnitude
#                 idx_val_exp = [
#                     (i, self.last_features[i], self.last_explanation[i])
#                     for i in range(len(self.last_features))
#                 ]
#                 top_k = sorted(
#                     idx_val_exp,
#                     key=lambda x: abs(x[2]),
#                     reverse=True
#                 )[:self.k]
#             else:
#                 # no explanation => store all
#                 top_k = [
#                     (i, self.last_features[i], None)
#                     for i in range(len(self.last_features))
#                 ]

#             for (i, val, exp_val) in top_k:
#                 feat_arr = [None] * len(self.last_features)
#                 feat_arr[i] = val

#                 self.memory.store_exemplar(
#                     label_probs=self.last_inference_probs,
#                     features=feat_arr,
#                     explanation=None,
#                     time_stored=current_time
#                 )

#             # Optional: you can add time here too if you want storage to incur cost
#             self.time.add_time(self.constant_time)

#         self.last_inference_probs = None
#         self.last_features = None
#         self.last_explanation = None

#     def infer(self, ui):
#         """
#         Perform one inference and return (probs, total_time).

#         - If explanation_type == 'attribution' and explanation is present:
#             Skip memory; sum explanation (top-k) directly.
#         - Else:
#             Use memory and importance-based or imputed importance logic.
#         """
#         start_time = self.time.get_time()

#         features = ui.get_value('features')
#         explanation = ui.get_value('explanation')

#         self.last_features = features
#         self.last_explanation = explanation

#         # Case 1: Pure attribution mode, explanation given:
#         # sum top-k explanation values directly and pass through logistic.
#         if self.explanation_type == 'attribution' and explanation is not None:
#             top_k_indices = np.argsort(np.abs(explanation))[-self.k:]
#             total_attr = sum(explanation[i] for i in top_k_indices)

#             prob_label_1 = 1.0 / (1.0 + math.exp(-self.scaling_factor * total_attr))
#             prob_label_0 = 1.0 - prob_label_1
#             probs = {0: prob_label_0, 1: prob_label_1}

#             self.last_inference_probs = probs
#             self.time.add_time(self.constant_time)
#             total_infer_time = self.time.get_time() - start_time
#             return probs, total_infer_time

#         # Case 2: Use memory (importance explanations or imputation)
#         total_attribution = 0.0
#         current_time = self.time.get_time()

#         # 2a) Explanation present, explanation_type == 'importance'
#         if explanation is not None and self.explanation_type == 'importance':
#             idx_val_exp = [(i, features[i], explanation[i]) for i in range(len(features))]
#             top_k_sorted = sorted(idx_val_exp, key=lambda x: abs(x[2]), reverse=True)

#             # Dropoff: keep only items where each next is >= 0.1× previous
#             filtered = []
#             for i, item in enumerate(top_k_sorted):
#                 if i == 0 or abs(item[2]) >= 0.1 * abs(top_k_sorted[i - 1][2]):
#                     filtered.append(item)
#                 else:
#                     break

#             top_k = filtered

#             for (i, feat_val, exp_val) in top_k:
#                 stored_exemplars = self.memory.retrieve_exemplars(
#                     current_time=current_time
#                 )

#                 local_strengths = {}
#                 for (ex, activation) in stored_exemplars:
#                     if i < len(ex['features']) and ex['features'][i] is not None:
#                         dist = abs(feat_val - ex['features'][i])
#                         sim = math.exp(-self.sensitivity * dist + activation)
#                         for lbl, p_lbl in ex['label_probs'].items():
#                             local_strengths[lbl] = local_strengths.get(lbl, 0.0) + sim * p_lbl

#                 if not local_strengths:
#                     continue

#                 local_dist = normalize_label_strengths(local_strengths)
#                 raw_sign = sum(
#                     (+1.0 if lbl == 1 else -1.0) * p_lbl
#                     for lbl, p_lbl in local_dist.items()
#                 )

#                 if abs(raw_sign) < 0.01:
#                     continue

#                 expected_sign = 1.0 if raw_sign > 0 else -1.0
#                 partial_attr = expected_sign * exp_val
#                 total_attribution += partial_attr

#         # 2b) No explanation: impute importance by similarity-weighted averaging
#         elif explanation is None and features is not None:
#             for i, feat_val in enumerate(features):
#                 stored_exemplars = self.memory.retrieve_exemplars(
#                     current_time=current_time
#                 )

#                 sim_weighted_sum = 0.0
#                 total_sim = 0.0

#                 for ex, activation in stored_exemplars:
#                     if i < len(ex['features']) and ex['features'][i] is not None:
#                         ex_val = ex['features'][i]
#                         dist = abs(feat_val - ex_val)
#                         sim = math.exp(-self.sensitivity * dist + activation)

#                         label_probs = ex['label_probs']
#                         label = max(label_probs, key=label_probs.get)
#                         label_sign = 1.0 if label == 1 else -1.0

#                         explanation_vector = ex.get("explanation")
#                         if explanation_vector is None or i >= len(explanation_vector):
#                             continue

#                         raw_exp_val = explanation_vector[i]

#                         # Flip sign if label and explanation disagree
#                         exp_val = raw_exp_val
#                         if math.copysign(1, raw_exp_val) != label_sign:
#                             exp_val *= -1

#                         sim_weighted_sum += sim * exp_val
#                         total_sim += sim

#                 if total_sim > 0:
#                     imputed_importance = sim_weighted_sum / total_sim
#                 else:
#                     imputed_importance = 0.0

#                 total_attribution += imputed_importance

#         # If we reach here and total_attribution is still 0, the logistic will be 0.5/0.5.
#         prob_label_1 = 1.0 / (1.0 + math.exp(-total_attribution * self.scaling_factor))
#         prob_label_0 = 1.0 - prob_label_1
#         probs = {0: prob_label_0, 1: prob_label_1}
#         self.last_inference_probs = probs

#         # Single constant time for inference
#         self.time.add_time(self.constant_time)
#         total_infer_time = self.time.get_time() - start_time
#         return probs, total_infer_time

#     def feedback(self, ui):
#         """
#         Feedback updates memory with single-feature exemplars.

#         - If explanation_type == 'importance':
#             Use ai_prediction as label for all stored exemplars.
#         - If explanation_type == 'attribution':
#             Label each stored exemplar by sign of its explanation
#             (+ => label=1, - => label=0).
#         """
#         start_time = self.time.get_time()

#         features = ui.get_value('features')
#         explanation = ui.get_value('explanation')
#         ai_pred = ui.get_value('ai_prediction')

#         if self.explanation_type == 'importance':
#             label_probs_global = {ai_pred: 1.0}
#         else:
#             label_probs_global = None  # per-feature in attribution mode

#         current_time = self.time.get_time()

#         # Determine top-k (with dropoff) if explanation present
#         if explanation is not None:
#             idx_val_exp = [(i, features[i], explanation[i]) for i in range(len(features))]
#             top_k_sorted = sorted(idx_val_exp, key=lambda x: abs(x[2]), reverse=True)

#             filtered = []
#             for i, item in enumerate(top_k_sorted):
#                 if i == 0 or abs(item[2]) >= 0.1 * abs(top_k_sorted[i - 1][2]):
#                     filtered.append(item)
#                 else:
#                     break

#             top_k = filtered
#         else:
#             top_k = [(i, features[i], None) for i in range(len(features))]

#         for (i, val, exp_val) in top_k:
#             feat_arr = [None] * len(features)
#             feat_arr[i] = val

#             if self.explanation_type == 'attribution':
#                 if exp_val is not None and exp_val >= 0:
#                     local_label_probs = {1: 1.0}
#                 else:
#                     local_label_probs = {0: 1.0}
#             else:
#                 local_label_probs = label_probs_global

#             self.memory.store_exemplar(
#                 label_probs=local_label_probs,
#                 features=feat_arr,
#                 explanation=explanation,
#                 time_stored=current_time
#             )
#             self.all_labels.update(local_label_probs.keys())

#         self.last_inference_probs = None
#         self.last_features = None
#         self.last_explanation = None

#         # Single constant time for feedback
#         self.time.add_time(self.constant_time)
#         total_time = self.time.get_time() - start_time
#         return total_time


# class AttributionSum(HumanModelBase):
#     """
#     Instead of a specialized SingleFeatureMemory, we use MemoryBase multiple times,
#     storing one "exemplar" per feature dimension. Each chunk has 'features' that
#     is effectively just [None, None, ..., value_at_idx, ..., None].
#     We then sum partial contributions.

#     New parameter: `explanation_type` can be:
#       - 'importance' (original behavior): 
#           feedback uses ai_prediction as the label.
#       - 'attribution':
#           feedback uses the sign of each explanation dimension
#           (+ => label=1, - => label=0).
#     """

#     def __init__(
#         self,
#         time=None,
#         decay_param=0.5,
#         retrieval_threshold=-0.3,
#         sensitivity=15.0,
#         scaling_factor=1.0,
#         save_new_exemplar=True,
#         k=2,
#         encoding_time_per_feature=0.0,
#         base_decision_time=2.0,
#         time_scale=0.0,
#         explanation_type='importance',  # <--- NEW PARAM
#         **kwargs
#     ):
#         super().__init__(time=time, decay_param=decay_param)
#         self.memory = MemoryBase(decay_param=decay_param, retrieval_threshold=retrieval_threshold)

#         self.sensitivity = sensitivity
#         self.scaling_factor = scaling_factor
#         self.save_new_exemplar = save_new_exemplar
#         self.k = k

#         # timing
#         self.encoding_time_per_feature = encoding_time_per_feature
#         self.base_decision_time = base_decision_time
#         self.time_scale = time_scale

#         # new
#         self.explanation_type = explanation_type

#     def new_instance(self):
#         """
#         If last_inference_probs + last_features => store top-k single-feature exemplars
#         (soft-labeled).
#         Each stored 'exemplar' has all features = None except the one dimension i.
#         """
#         if self.last_inference_probs is not None and self.last_features is not None:
#             if self.save_new_exemplar:
#                 current_time = self.time.get_time() if self.time else 0.0

#                 if self.last_explanation is not None:
#                     # pick top-k by explanation magnitude
#                     idx_val_exp = [(i, self.last_features[i], self.last_explanation[i])
#                                    for i in range(len(self.last_features))]
#                     top_k = sorted(idx_val_exp, key=lambda x: abs(x[2]), reverse=True)[:self.k]
#                 else:
#                     # no explanation => store them all
#                     top_k = [(i, self.last_features[i], None) 
#                              for i in range(len(self.last_features))]

#                 for (i, val, exp_val) in top_k:
#                     feat_arr = [None]*len(self.last_features)
#                     feat_arr[i] = val

#                     self.memory.store_exemplar(
#                         label_probs=self.last_inference_probs,
#                         features=feat_arr,
#                         explanation=None,
#                         time_stored=current_time
#                     )

#                 self.time.add_time(self.constant_feedback_time)    

#         self.last_inference_probs = None
#         self.last_features = None
#         self.last_explanation = None

#     def infer(self, ui, debug=False):
#         """
#         1) encoding time
#         2) For each feature dimension i, retrieve relevant exemplars from memory
#         3) sum partial attributions => logistic
#         4) EBRW-based decision time
#         """
#         start_time = self.time.get_time()

#         features = ui.get_value('features')
#         explanation = ui.get_value('explanation')

#         # 1) encoding time
#         if not explanation:
#             n_features = len(features) if features else 0
#             encoding_time = n_features * self.encoding_time_per_feature
#         else:
#             n_features = self.k
#             encoding_time = n_features * self.encoding_time_per_feature + len(explanation) * self.encoding_time_per_feature

#         if self.time:
#             self.time.add_time(encoding_time)

#         self.last_features = features
#         self.last_explanation = explanation

#         # If explanation_type is 'attribution', skip memory and sum raw explanation
#         if self.explanation_type == 'attribution' and explanation:
#             # sum only the top-k explanation values based on absolute value
#             top_k_indices = np.argsort(np.abs(explanation))[-self.k:]
#             total_attr = sum(explanation[i] for i in top_k_indices)
#             # total_attr = sum(explanation)
#             prob_label_1 = 1.0 / (1.0 + math.exp(-self.scaling_factor * total_attr))
#             prob_label_0 = 1.0 - prob_label_1
#             probs = {0: prob_label_0, 1: prob_label_1}

#             dist_to_boundary = abs(prob_label_1 - 0.5)
#             decision_time = self.base_decision_time + self.time_scale * (1 - 2 * dist_to_boundary)
#             decision_time = max(decision_time, self.base_decision_time)

#             total_infer_time = encoding_time + decision_time
#             if self.time:
#                 self.time.add_time(decision_time)

#             self.last_inference_probs = probs
#             return probs, total_infer_time

#         total_attribution = 0.0
#         current_time = self.time.get_time() if self.time else 0.0

#         # If explanation is available and type is 'importance', pick top-k with dropoff
#         if explanation is not None and self.explanation_type == 'importance':
#             idx_val_exp = [(i, features[i], explanation[i]) for i in range(len(features))]
#             top_k_sorted = sorted(idx_val_exp, key=lambda x: abs(x[2]), reverse=True)

#             filtered = []
#             for i, item in enumerate(top_k_sorted):
#                 if i == 0 or abs(item[2]) >= 0.1 * abs(top_k_sorted[i - 1][2]):
#                     filtered.append(item)
#                 else:
#                     break

#             top_k = filtered

#             for (i, feat_val, exp_val) in top_k:
#                 stored_exemplars = self.memory.retrieve_exemplars(
#                     current_time=current_time
#                 )

#                 if debug:
#                     print(f"Feature {i}: {feat_val}, Explanation Value: {exp_val}")

#                 local_strengths = {}
#                 for (ex, activation) in stored_exemplars:
#                     if i < len(ex['features']) and ex['features'][i] is not None:
#                         dist = abs(feat_val - ex['features'][i])
#                         sim = math.exp(-self.sensitivity * dist + activation)
#                         for lbl, p_lbl in ex['label_probs'].items():
#                             local_strengths[lbl] = local_strengths.get(lbl, 0.0) + sim * p_lbl
#                         if debug:
#                             print(f"Exemplar {ex['features']}, Label: {ex['label_probs']}, similarity: {sim}")

#                 if not local_strengths:
#                     continue

#                 local_dist = normalize_label_strengths(local_strengths)
#                 raw_sign = sum((+1.0 if lbl == 1 else -1.0) * p_lbl for lbl, p_lbl in local_dist.items())

#                 if abs(raw_sign) < 0.01:
#                     continue

#                 expected_sign = 1.0 if raw_sign > 0 else -1.0
#                 partial_attr = expected_sign * exp_val
#                 total_attribution += partial_attr

#                 retrieval_time = 0.4
#                 if self.time:
#                     self.time.add_time(retrieval_time)

#         # If no explanation, use similarity-weighted imputation for each feature
#         elif explanation is None:
#             for i, feat_val in enumerate(features):
#                 stored_exemplars = self.memory.retrieve_exemplars(
#                     current_time=current_time
#                 )

#                 sim_weighted_sum = 0.0
#                 total_sim = 0.0

#                 for ex, activation in stored_exemplars:
#                     if i < len(ex['features']) and ex['features'][i] is not None:
#                         ex_val = ex['features'][i]
#                         dist = abs(feat_val - ex_val)
#                         sim = math.exp(-self.sensitivity * dist + activation)

#                         label_probs = ex['label_probs']
#                         label = max(label_probs, key=label_probs.get)
#                         label_sign = 1.0 if label == 1 else -1.0

#                         explanation_vector = ex.get("explanation")
#                         if explanation_vector is None or i >= len(explanation_vector):
#                             continue

#                         raw_exp_val = explanation_vector[i]

#                         # Flip the sign if label and explanation disagree
#                         exp_val = raw_exp_val
#                         if math.copysign(1, raw_exp_val) != label_sign:
#                             exp_val *= -1

#                         sim_weighted_sum += sim * exp_val
#                         total_sim += sim

#                         if debug:
#                             print(f"Exemplar {ex['features']}, Label: {label_probs}, "
#                                   f"Similarity: {sim}, Explanation Value: {exp_val}")

#                 if total_sim > 0:
#                     imputed_importance = sim_weighted_sum / total_sim
#                 else:
#                     imputed_importance = 0.0

#                 total_attribution += imputed_importance

#                 retrieval_time = 0.4
#                 if self.time:
#                     self.time.add_time(retrieval_time)

#                 if debug:
#                     print(f"Feature {i}: {feat_val}, Imputed Importance: {imputed_importance}")

#             if debug:
#                 print(f"Total Attribution: {total_attribution}")
#         # 3) Logistic to get probabilities
#         prob_label_1 = 1.0 / (1.0 + math.exp(-total_attribution * self.scaling_factor))
#         prob_label_0 = 1.0 - prob_label_1
#         probs = {0: prob_label_0, 1: prob_label_1}
#         self.last_inference_probs = probs

#         # 4) Decision time
#         decision_time = compute_decision_time(probs, self.base_decision_time, self.time_scale)
#         total_infer_time = encoding_time + decision_time
#         if self.time:
#             self.time.add_time(decision_time)

#         total_infer_time = self.time.get_time() - start_time

#         return probs, total_infer_time

#     def feedback(self, ui):
#         """
#         If `explanation_type` == 'importance', the label from ai_prediction is used (original code).
#         If `explanation_type` == 'attribution', we determine the label from the sign of explanation
#            (+ => label=1, - => label=0) for each feature in top_k.
#         """ 
#         features = ui.get_value('features')
#         explanation = ui.get_value('explanation')
#         ai_pred = ui.get_value('ai_prediction')

#         # If "importance", same as original
#         if self.explanation_type == 'importance':
#             label_probs = {ai_pred: 1.0}
#         else:
#             label_probs = None  # Will be set inside the loop if "attribution"

#         current_time = self.time.get_time() if self.time else 0.0

#         if explanation is not None:
#             idx_val_exp = [(i, features[i], explanation[i]) for i in range(len(features))]
#             top_k_sorted = sorted(idx_val_exp, key=lambda x: abs(x[2]), reverse=True)

#             # Keep only the top items where each next is >= 0.1× previous
#             filtered = []
#             for i, item in enumerate(top_k_sorted):
#                 if i == 0 or abs(item[2]) >= 0.1 * abs(top_k_sorted[i - 1][2]):
#                     filtered.append(item)
#                 else:
#                     break

#             top_k = filtered
#         else:
#             top_k = [(i, features[i], None) for i in range(len(features))]

#         for (i, val, exp_val) in top_k:
#             feat_arr = [None]*len(features)
#             feat_arr[i] = val

#             # If "attribution", we choose label by sign:
#             if self.explanation_type == 'attribution':
#                 if exp_val is not None and exp_val >= 0:
#                     local_label_probs = {1: 1.0}
#                 else:
#                     local_label_probs = {0: 1.0}
#             else:
#                 # "importance" => use the same label_probs for all top-k
#                 local_label_probs = label_probs

#             self.memory.store_exemplar(
#                 label_probs=local_label_probs,
#                 features=feat_arr,
#                 explanation=explanation,
#                 time_stored=current_time
#             )
#             # Also store the label in self.all_labels
#             self.all_labels.update(local_label_probs.keys())

#         self.last_inference_probs = None
#         self.last_features = None
#         self.last_explanation = None

#         if self.time:
#             self.time.add_time(self.constant_feedback_time)
#         return self.constant_feedback_time



# Need to test this updated code