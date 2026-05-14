"""
Abstract reasoning strategy interface for cognitive agent reasoning engines.

This module defines the contract that all reasoning strategies (CoAX, CoXAM forward,
CoXAM counterfactual) must implement, enabling unified strategy orchestration.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, Type
from enum import Enum


class StrategyType(Enum):
    """Reasoning strategy classification."""
    COAX_FORWARD = "coax_forward"              # CoAX forward reasoning
    COXAM_FORWARD = "coxam_forward"            # CoXAM forward reasoning
    COXAM_COUNTERFACTUAL = "coxam_counterfactual"  # CoXAM counterfactual reasoning


class ReasoningMode(Enum):
    """Operation mode for strategies."""
    RETRIEVE = "retrieve"                      # Use memory-based probabilistic retrieval
    READ = "read"                              # Direct/deterministic read from model
    HEURISTIC = "heuristic"                    # Simplified heuristic reasoning


@dataclass
class StrategyConfig:
    """Configuration parameters for strategy instantiation."""
    
    # Common parameters
    strategy_name: str
    strategy_type: StrategyType
    mode: ReasoningMode = ReasoningMode.RETRIEVE
    
    # Cognitive parameters
    decay_param: float = 0.5
    retrieval_threshold: float = -2.5
    sensitivity: float = 10.0
    
    # Temporal parameters
    time_manager: Optional[Any] = None  # Time tracking object (tick, get_time, add_time)
    
    # Additional parameters (strategy-specific)
    extra_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.extra_params is None:
            self.extra_params = {}


@dataclass
class StrategyMetadata:
    """Metadata describing a reasoning strategy."""
    
    name: str                          # Unique strategy identifier
    display_name: str                  # Human-readable name
    strategy_type: StrategyType        # Category
    description: str                   # What this strategy does
    category: str                      # "CoAX", "CoXAM", etc.
    
    supported_modes: list = None       # Modes this strategy supports
    parameters: Dict[str, Any] = None  # Parameter hints and defaults
    
    def __post_init__(self):
        if self.supported_modes is None:
            self.supported_modes = [ReasoningMode.RETRIEVE]
        if self.parameters is None:
            self.parameters = {}


class ReasoningStrategy(ABC):
    """
    Abstract base class for all reasoning strategies.
    
    Any reasoning strategy (whether CoAX-based, CoXAM-based, or hybrid) must
    implement this interface to be compatible with the unified strategy registry
    and orchestration system.
    
    Core responsibilities:
      1. Initialization with config (parameters, memory, time tracking)
      2. Instance management (new_instance for trial boundaries)
      3. Inference (make predictions/decisions)
      4. Feedback (learn from outcomes)
    """
    
    @property
    @abstractmethod
    def metadata(self) -> StrategyMetadata:
        """Return strategy metadata (name, description, capabilities)."""
        pass
    
    @abstractmethod
    def new_instance(self):
        """
        Signal start of a new trial/instance.
        
        Called at the beginning of each new decision task. Subclasses should use
        this to finalize learning from previous trial and reset trial-specific state.
        """
        pass
    
    @abstractmethod
    def infer(self, features: Dict[str, Any], explanation: Optional[Any] = None,
              ai_prediction: Optional[int] = None, **kwargs) -> Tuple[Dict[int, float], float, Dict[str, Any]]:
        """
        Make a prediction/decision based on input features and optional explanation.
        
        Args:
            features: Input feature vector (can be np.ndarray, list, or dict)
            explanation: Optional explanation vector (salience, attribution, etc.)
            ai_prediction: Optional AI model's prediction (for comparison)
            **kwargs: Additional strategy-specific parameters
            
        Returns:
            (probabilities, time_cost, info):
              - probabilities: {label -> probability} distribution over choices
              - time_cost: Float seconds representing cognitive time used
              - info: Dict with diagnostic/debug information
                  * 'activated_exemplars': Count of memory retrievals
                  * 'focus_features': Which features were attended to
                  * 'basis': Reasoning basis (e.g., "top-3 exemplars", "aggregated")
                  * other strategy-specific debug info
        """
        pass
    
    @abstractmethod
    def feedback(self, features: Dict[str, Any], true_label: int, 
                 explanation: Optional[Any] = None, **kwargs) -> float:
        """
        Process feedback/learning signal (e.g., correct label revealed).
        
        Called after a decision to enable learning from correctness feedback.
        Strategies use this to update memory, reinforce useful patterns, etc.
        
        Args:
            features: Feature vector of this trial
            true_label: Ground truth label (0, 1, etc.)
            explanation: Optional explanation (may be used for learning)
            **kwargs: Additional strategy-specific parameters
            
        Returns:
            float: Cognitive time cost for processing feedback
        """
        pass
    
    def get_state(self) -> Dict[str, Any]:
        """
        Export strategy state for persistence/debugging.
        
        Optional override. Default implementation returns empty dict.
        Subclasses should override to enable state inspection.
        
        Returns:
            Dict with:
              - 'memory_size': Number of exemplars/chunks stored
              - 'exemplars_count': Total stored items
              - 'last_inference': Last prediction details
              - other debug info
        """
        return {}
    
    def set_state(self, state: Dict[str, Any]):
        """
        Restore strategy state from dict.
        
        Optional override. Default implementation is no-op.
        
        Args:
            state: State dict (format defined by get_state())
        """
        pass


class CounterfactualStrategy(ReasoningStrategy):
    """
    Extended interface for counterfactual reasoning strategies.
    
    Adds counterexample generation capability on top of forward reasoning.
    Strategies implementing this interface can suggest feature changes to
    alter a decision outcome.
    """
    
    @abstractmethod
    def suggest_change(self, features: Dict[str, Any],
                      bounds: Dict[str, Tuple[float, float]],
                      desired_outcome: int,
                      **kwargs) -> Dict[str, Dict[str, float]]:
        """
        Suggest feature changes to reach a different decision outcome.
        
        Args:
            features: Current feature vector
            bounds: Per-feature (min, max) value bounds
            desired_outcome: Target outcome (0, 1, etc.)
            **kwargs: Strategy-specific parameters
            
        Returns:
            Dict mapping features to change info:
              {
                'feature_name': {
                  'probability': float,       # Expected probability of selecting this feature
                  'mean_change': float,       # Mean required change magnitude
                  'expected_time': float      # Cognitive time for this change
                },
                ...
              }
        """
        pass
