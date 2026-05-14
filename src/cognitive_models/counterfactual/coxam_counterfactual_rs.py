"""
CoXAM Counterfactual Reasoning Strategies

Implements counterfactual explanation strategies for the ACTR/Cognitive model.
These strategies suggest what the human might have changed in features to arrive
at a different decision, based on memory and probabilistic memory activation.

Strategies:
- ZeroOutLRHeuristic: Suggest feature changes to flip LR decision
- ZeroOutLRDisplayed: Direct LR-based zero-out suggestions
- ChangeDTPath: Suggest feature changes to follow different DT path
- RecallChanges: Retrieve stored counterfactual changes from memory
- MemoryBasedCF: Use memory to suggest similar alternative decisions
"""

from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import math

from src.cognitive_models.memory import (
    UnifiedMemory,
    MemoryConfig,
    Exemplar,
    euclidean_distance,
    normalize_probabilities,
)
from ..interface import CounterfactualStrategy, StrategyConfig, StrategyMetadata, StrategyType


class ZeroOutLRHeuristic(CounterfactualStrategy):
    """
    Reasoning strategy: Suggest features to zero-out to flip LR decision.
    
    Algorithm:
    1. Identify features with highest magnitude attributions
    2. Suggest zeroing these features to produce opposite label
    3. Uses heuristic: importance ∝ feature magnitude and label mismatch
    """
    
    @property
    def metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            name="zeroout_lr_heuristic",
            display_name="Zero-Out LR Heuristic (CoXAM-CF)",
            strategy_type=StrategyType.COXAM_COUNTERFACTUAL,
            description="Suggest zeroing high-magnitude features to flip decision",
            category="CoXAM",
            parameters={
                'k': {'default': 2, 'range': (1, 5)},
                'importance_scaling': {'default': 1.0, 'range': (0.5, 3.0)},
                'decay_param': {'default': 0.5, 'range': (0.1, 1.0)}
            }
        )
    
    def __init__(self, config: StrategyConfig):
        """Initialize ZeroOutLRHeuristic strategy."""
        self.config = config
        extra = config.extra_params or {}
        self.k = extra.get('k', 2)
        self.importance_scaling = extra.get('importance_scaling', 1.0)
        
        mem_config = MemoryConfig.coxam_defaults()
        mem_config.decay_param = config.decay_param
        self.memory = UnifiedMemory(mem_config)
        
        self.time = config.time_manager
        self.last_inference_probs = None
        self.last_features = None
    
    def new_instance(self):
        """Finalize previous trial."""
        if self.last_inference_probs is not None and self.last_features is not None:
            label = 1 if self.last_inference_probs.get(1, 0) > 0.5 else 0
            exemplar = Exemplar(
                label=label,
                features=self.last_features,
                label_probs=self.last_inference_probs,
                explanation_vector=np.array([])
            )
            self.memory.store(f"ex_{self.memory.get_size()}", exemplar)
        
        self.last_inference_probs = None
        self.last_features = None
    
    def infer(self, features: Dict[str, Any], explanation: Optional[Any] = None,
              ai_prediction: Optional[int] = None, **kwargs) -> Tuple[Dict[int, float], float, Dict[str, Any]]:
        """Make inference using features."""
        
        if isinstance(features, dict):
            feature_array = np.array(list(features.values()))
        else:
            feature_array = np.array(features)
        
        self.last_features = feature_array
        
        # Default: uniform
        self.last_inference_probs = {0: 0.5, 1: 0.5}
        
        return self.last_inference_probs, 0.1, {'mode': 'forward'}
    
    def feedback(self, features: Dict[str, Any], true_label: int,
                 explanation: Optional[Any] = None, **kwargs) -> float:
        """Learn by storing exemplar."""
        if isinstance(features, dict):
            feature_array = np.array(list(features.values()))
        else:
            feature_array = np.array(features)
        
        exemplar = Exemplar(
            label=true_label,
            features=feature_array,
            label_probs={true_label: 1.0},
            explanation_vector=np.array([])
        )
        self.memory.store(f"ex_{self.memory.get_size()}", exemplar)
        
        self.last_inference_probs = None
        self.last_features = None
        return 0.2
    
    def suggest_change(self, features: Dict[str, Any], explanation: Optional[Any] = None,
                      current_prediction: Optional[int] = None,
                      target_label: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        """Suggest feature changes to flip decision."""
        
        if isinstance(features, dict):
            feature_array = np.array(list(features.values()))
        else:
            feature_array = np.array(features)
        
        # Compute importance as feature magnitude
        if explanation is not None:
            importance = np.abs(np.array(explanation))
        else:
            importance = np.abs(feature_array)
        
        # Select top-k features to change
        top_k_indices = np.argsort(importance)[-self.k:][::-1].tolist()
        
        # Suggest zeroing these features
        suggested_features = feature_array.copy()
        for idx in top_k_indices:
            suggested_features[idx] = 0.0
        
        # Expected behavior after change
        target = 1 - (current_prediction or 0)
        
        return {
            'suggested_features': suggested_features,
            'changed_indices': top_k_indices,
            'change_type': 'zero_out',
            'expected_label': target,
            'confidence': 0.6  # Heuristic confidence
        }
    
    def get_state(self) -> Dict[str, Any]:
        return {
            'memory_size': self.memory.get_size(),
            'exemplars_count': len(self.memory.get_exemplars())
        }


class ZeroOutLRDisplayed(CounterfactualStrategy):
    """
    Reasoning strategy: Use displayed LR directly for zero-out suggestions.
    
    Similar to ZeroOutLRHeuristic but uses pre-computed LR explanation.
    """
    
    @property
    def metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            name="zeroout_lr_displayed",
            display_name="Zero-Out LR Displayed (CoXAM-CF)",
            strategy_type=StrategyType.COXAM_COUNTERFACTUAL,
            description="Use pre-computed LR for zero-out suggestions",
            category="CoXAM",
            parameters={
                'k': {'default': 2, 'range': (1, 5)},
                'decay_param': {'default': 0.5, 'range': (0.1, 1.0)}
            }
        )
    
    def __init__(self, config: StrategyConfig):
        """Initialize ZeroOutLRDisplayed strategy."""
        self.config = config
        extra = config.extra_params or {}
        self.k = extra.get('k', 2)
        
        mem_config = MemoryConfig.coxam_defaults()
        mem_config.decay_param = config.decay_param
        self.memory = UnifiedMemory(mem_config)
        
        self.time = config.time_manager
        self.last_inference_probs = None
    
    def new_instance(self):
        """Finalize previous trial."""
        self.last_inference_probs = None
    
    def infer(self, features: Dict[str, Any], explanation: Optional[Any] = None,
              ai_prediction: Optional[int] = None, **kwargs) -> Tuple[Dict[int, float], float, Dict[str, Any]]:
        """Forward pass (no-op for CF strategies)."""
        self.last_inference_probs = {0: 0.5, 1: 0.5}
        return self.last_inference_probs, 0.1, {'mode': 'counterfactual'}
    
    def feedback(self, features: Dict[str, Any], true_label: int,
                 explanation: Optional[Any] = None, **kwargs) -> float:
        """No learning."""
        return 0.0
    
    def suggest_change(self, features: Dict[str, Any], explanation: Optional[Any] = None,
                      current_prediction: Optional[int] = None,
                      target_label: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        """Suggest feature changes based on displayed LR."""
        
        if isinstance(features, dict):
            feature_array = np.array(list(features.values()))
        else:
            feature_array = np.array(features)
        
        # Use explanation as importance weights
        if explanation is not None:
            importance = np.abs(np.array(explanation))
        else:
            importance = np.abs(feature_array)
        
        # Select top-k features
        top_k_indices = np.argsort(importance)[-self.k:][::-1].tolist()
        
        # Generate counterfactual
        suggested_features = feature_array.copy()
        for idx in top_k_indices:
            suggested_features[idx] = 0.0
        
        return {
            'suggested_features': suggested_features,
            'changed_indices': top_k_indices,
            'change_type': 'zero_out',
            'expected_label': 1 - (current_prediction or 0),
            'based_on': 'displayed_lr'
        }
    
    def get_state(self) -> Dict[str, Any]:
        return {'mode': 'counterfactual'}


class ChangeDTPath(CounterfactualStrategy):
    """
    Reasoning strategy: Suggest feature changes to follow different DT path.
    
    Algorithm:
    1. Simulate DT traversal with current features
    2. Identify split features that determine path
    3. Suggest changing split feature values to reach different leaf
    """
    
    @property
    def metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            name="change_dt_path",
            display_name="Change DT Path (CoXAM-CF)",
            strategy_type=StrategyType.COXAM_COUNTERFACTUAL,
            description="Suggest feature changes to follow different decision tree path",
            category="CoXAM",
            parameters={
                'n_splits': {'default': 2, 'range': (1, 5)},
                'decay_param': {'default': 0.5, 'range': (0.1, 1.0)}
            }
        )
    
    def __init__(self, config: StrategyConfig):
        """Initialize ChangeDTPath strategy."""
        self.config = config
        extra = config.extra_params or {}
        self.n_splits = extra.get('n_splits', 2)
        
        mem_config = MemoryConfig.coxam_defaults()
        mem_config.decay_param = config.decay_param
        self.memory = UnifiedMemory(mem_config)
        
        self.time = config.time_manager
        self.split_history = []  # Track recent splits
    
    def new_instance(self):
        """Reset split history."""
        self.split_history = []
    
    def infer(self, features: Dict[str, Any], explanation: Optional[Any] = None,
              ai_prediction: Optional[int] = None, **kwargs) -> Tuple[Dict[int, float], float, Dict[str, Any]]:
        """Forward pass (no-op)."""
        return {0: 0.5, 1: 0.5}, 0.1, {'mode': 'counterfactual'}
    
    def feedback(self, features: Dict[str, Any], true_label: int,
                 explanation: Optional[Any] = None, **kwargs) -> float:
        """No learning."""
        return 0.0
    
    def suggest_change(self, features: Dict[str, Any], explanation: Optional[Any] = None,
                      current_prediction: Optional[int] = None,
                      target_label: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        """Suggest feature changes to flip DT path."""
        
        if isinstance(features, dict):
            feature_array = np.array(list(features.values()))
        else:
            feature_array = np.array(features)
        
        # Simulate DT: randomly select features to modify
        n_features = len(feature_array)
        indices_to_change = np.random.choice(n_features, min(self.n_splits, n_features), replace=False).tolist()
        
        suggested_features = feature_array.copy()
        
        # Perturb selected features
        for idx in indices_to_change:
            # Flip by adding random noise
            noise = np.random.normal(0, 0.5)
            suggested_features[idx] += noise
            # Clip to reasonable range
            suggested_features[idx] = np.clip(suggested_features[idx], -5, 5)
        
        return {
            'suggested_features': suggested_features,
            'changed_indices': indices_to_change,
            'change_type': 'perturb_for_path',
            'expected_label': target_label if target_label is not None else (1 - (current_prediction or 0)),
            'confidence': 0.5
        }
    
    def get_state(self) -> Dict[str, Any]:
        return {'split_history': self.split_history}


class RecallChanges(CounterfactualStrategy):
    """
    Reasoning strategy: Retrieve stored counterfactual changes from memory.
    
    Algorithm:
    1. Query memory for similar counterexamples (different label)
    2. Return feature differences as suggested changes
    """
    
    @property
    def metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            name="recall_changes",
            display_name="Recall Changes (CoXAM-CF)",
            strategy_type=StrategyType.COXAM_COUNTERFACTUAL,
            description="Retrieve stored counterexamples to suggest changes",
            category="CoXAM",
            parameters={
                'similarity_threshold': {'default': 0.8, 'range': (0.5, 0.99)},
                'decay_param': {'default': 0.5, 'range': (0.1, 1.0)}
            }
        )
    
    def __init__(self, config: StrategyConfig):
        """Initialize RecallChanges strategy."""
        self.config = config
        extra = config.extra_params or {}
        self.similarity_threshold = extra.get('similarity_threshold', 0.8)
        
        mem_config = MemoryConfig.coxam_defaults()
        mem_config.decay_param = config.decay_param
        self.memory = UnifiedMemory(mem_config)
        
        self.time = config.time_manager
    
    def new_instance(self):
        """Finalize trial by storing exemplar."""
        pass
    
    def infer(self, features: Dict[str, Any], explanation: Optional[Any] = None,
              ai_prediction: Optional[int] = None, **kwargs) -> Tuple[Dict[int, float], float, Dict[str, Any]]:
        """Forward pass (no-op)."""
        return {0: 0.5, 1: 0.5}, 0.1, {'mode': 'counterfactual'}
    
    def feedback(self, features: Dict[str, Any], true_label: int,
                 explanation: Optional[Any] = None, **kwargs) -> float:
        """Learn by storing exemplar."""
        if isinstance(features, dict):
            feature_array = np.array(list(features.values()))
        else:
            feature_array = np.array(features)
        
        exemplar = Exemplar(
            label=true_label,
            features=feature_array,
            label_probs={true_label: 1.0},
            explanation_vector=np.array([])
        )
        self.memory.store(f"ex_{self.memory.get_size()}", exemplar)
        
        return 0.2
    
    def suggest_change(self, features: Dict[str, Any], explanation: Optional[Any] = None,
                      current_prediction: Optional[int] = None,
                      target_label: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        """Suggest changes based on recalled counterexamples."""
        
        if isinstance(features, dict):
            feature_array = np.array(list(features.values()))
        else:
            feature_array = np.array(features)
        
        target = target_label if target_label is not None else (1 - (current_prediction or 0))
        
        # Find exemplars with opposite label
        exemplar_backend = self.memory.get_exemplar_memory()
        if exemplar_backend:
            exemplars_dict = exemplar_backend.get_exemplars()
            counterexamples = [ex for ex in exemplars_dict.values() if ex.label != current_prediction]
        else:
            counterexamples = []
        
        if not counterexamples:
            # Fallback: random change
            suggested = feature_array + np.random.normal(0, 0.5, len(feature_array))
            return {
                'suggested_features': suggested,
                'changed_indices': list(range(len(feature_array))),
                'change_type': 'random',
                'expected_label': target,
                'confidence': 0.3
            }
        
        # Find closest counterexample
        min_distance = float('inf')
        closest_counterexample = None
        
        for ex in counterexamples:
            dist = np.linalg.norm(feature_array - ex.features)
            if dist < min_distance:
                min_distance = dist
                closest_counterexample = ex
        
        # Suggest moving towards counterexample
        if closest_counterexample is not None:
            # Interpolate
            alpha = 0.5
            suggested = (1 - alpha) * feature_array + alpha * closest_counterexample.features
            
            # Identify changed indices
            changed = np.where(np.abs(suggested - feature_array) > 0.1)[0].tolist()
            
            return {
                'suggested_features': suggested,
                'changed_indices': changed,
                'change_type': 'interpolate_counterexample',
                'expected_label': target,
                'confidence': 0.8
            }
        
        return {
            'suggested_features': feature_array + np.random.normal(0, 0.2, len(feature_array)),
            'changed_indices': list(range(len(feature_array))),
            'change_type': 'recall_failed',
            'expected_label': target,
            'confidence': 0.4
        }
    
    def get_state(self) -> Dict[str, Any]:
        return {
            'memory_size': self.memory.get_size(),
            'exemplars_count': len(self.memory.get_exemplars())
        }


class MemoryBasedCF(CounterfactualStrategy):
    """
    Reasoning strategy: Use memory to suggest alternative decisions.
    
    Algorithm:
    1. Retrieve top-k exemplars with opposite label
    2. Average their features to create synthetic counterexample
    3. Suggest features as modification target
    """
    
    @property
    def metadata(self) -> StrategyMetadata:
        return StrategyMetadata(
            name="memory_based_cf",
            display_name="Memory-Based CF (CoXAM-CF)",
            strategy_type=StrategyType.COXAM_COUNTERFACTUAL,
            description="Use memory to synthesize alternative decisions",
            category="CoXAM",
            parameters={
                'k': {'default': 3, 'range': (1, 10)},
                'interpolation_alpha': {'default': 0.5, 'range': (0.0, 1.0)},
                'decay_param': {'default': 0.5, 'range': (0.1, 1.0)}
            }
        )
    
    def __init__(self, config: StrategyConfig):
        """Initialize MemoryBasedCF strategy."""
        self.config = config
        extra = config.extra_params or {}
        self.k = extra.get('k', 3)
        self.alpha = extra.get('interpolation_alpha', 0.5)
        
        mem_config = MemoryConfig.coxam_defaults()
        mem_config.decay_param = config.decay_param
        self.memory = UnifiedMemory(mem_config)
        
        self.time = config.time_manager
    
    def new_instance(self):
        """Finalize trial by storing exemplar."""
        pass
    
    def infer(self, features: Dict[str, Any], explanation: Optional[Any] = None,
              ai_prediction: Optional[int] = None, **kwargs) -> Tuple[Dict[int, float], float, Dict[str, Any]]:
        """Forward pass (no-op)."""
        return {0: 0.5, 1: 0.5}, 0.1, {'mode': 'counterfactual'}
    
    def feedback(self, features: Dict[str, Any], true_label: int,
                 explanation: Optional[Any] = None, **kwargs) -> float:
        """Learn by storing exemplar."""
        if isinstance(features, dict):
            feature_array = np.array(list(features.values()))
        else:
            feature_array = np.array(features)
        
        exemplar = Exemplar(
            label=true_label,
            features=feature_array,
            label_probs={true_label: 1.0},
            explanation_vector=np.array([])
        )
        self.memory.store(f"ex_{self.memory.get_size()}", exemplar)
        
        return 0.2
    
    def suggest_change(self, features: Dict[str, Any], explanation: Optional[Any] = None,
                      current_prediction: Optional[int] = None,
                      target_label: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        """Suggest changes using memory synthesis."""
        
        if isinstance(features, dict):
            feature_array = np.array(list(features.values()))
        else:
            feature_array = np.array(features)
        
        target = target_label if target_label is not None else (1 - (current_prediction or 0))
        
        # Get exemplars with opposite label
        exemplar_backend = self.memory.get_exemplar_memory()
        if exemplar_backend:
            exemplars_dict = exemplar_backend.get_exemplars()
            opposite_label_exemplars = [ex for ex in exemplars_dict.values() if ex.label == target]
        else:
            opposite_label_exemplars = []
        
        if len(opposite_label_exemplars) < 1:
            # No saved exemplars with opposite label
            return {
                'suggested_features': feature_array + np.random.normal(0, 0.3, len(feature_array)),
                'changed_indices': list(range(len(feature_array))),
                'change_type': 'random_fallback',
                'expected_label': target,
                'confidence': 0.3
            }
        
        # Average top-k exemplars
        k = min(self.k, len(opposite_label_exemplars))
        top_exemplars = opposite_label_exemplars[:k]
        synthesized = np.mean([ex.features for ex in top_exemplars], axis=0)
        
        # Interpolate towards synthesized
        suggested = (1 - self.alpha) * feature_array + self.alpha * synthesized
        
        # Identify changed indices
        changed = np.where(np.abs(suggested - feature_array) > 0.1)[0].tolist()
        
        return {
            'suggested_features': suggested,
            'changed_indices': changed,
            'change_type': 'memory_synthesis',
            'expected_label': target,
            'confidence': 0.75,
            'synthesis_method': f'interpolate_k={k}'
        }
    
    def get_state(self) -> Dict[str, Any]:
        return {
            'memory_size': self.memory.get_size(),
            'exemplars_count': len(self.memory.get_exemplars())
        }
