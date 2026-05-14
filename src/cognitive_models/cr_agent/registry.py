"""
CR Agent and Environment Registry

Registry-based systems for managing:
1. Available agents (policies and strategies)
2. Available environments
3. Pre-trained model metadata
4. Cognitive parameter presets

Allows dynamic registration and lookup without hardcoding relationships.
"""

from typing import Dict, Any, Optional, Type, Callable
from dataclasses import dataclass
from enum import Enum


class AgentType(Enum):
    """Types of agents available."""
    DT_POLICY = "dt_policy"
    LR_CALC_POLICY = "lr_calc_policy"
    LR_HEUR_POLICY = "lr_heur_policy"
    CF_STRATEGY_DT = "cf_strategy_dt"
    CF_STRATEGY_LR_HEUR = "cf_strategy_lr_heur"
    CF_STRATEGY_LR_DISPLAYED = "cf_strategy_lr_displayed"
    CF_STRATEGY_RECALL_DT = "cf_strategy_recall_dt"
    CF_STRATEGY_RECALL_LR = "cf_strategy_recall_lr"


class EnvironmentType(Enum):
    """Types of environments available."""
    META_ROUTER = "meta_router"
    DT_FORWARD = "dt_forward"
    COUNTERFACTUAL = "counterfactual"


@dataclass
class AgentMetadata:
    """Metadata for an agent."""
    agent_type: AgentType
    name: str
    description: str
    model_class: Optional[Type] = None
    default_model_path: Optional[str] = None
    hyperparams: Optional[Dict[str, Any]] = None


@dataclass
class EnvironmentMetadata:
    """Metadata for an environment."""
    env_type: EnvironmentType
    name: str
    description: str
    env_class: Optional[Type] = None


class AgentRegistry:
    """Registry for agent types and models."""
    
    _registry: Dict[AgentType, AgentMetadata] = {}
    
    @classmethod
    def register(cls, metadata: AgentMetadata) -> None:
        """Register an agent type."""
        cls._registry[metadata.agent_type] = metadata
    
    @classmethod
    def get(cls, agent_type: AgentType) -> Optional[AgentMetadata]:
        """Get metadata for agent type."""
        return cls._registry.get(agent_type)
    
    @classmethod
    def list_all(cls) -> Dict[AgentType, AgentMetadata]:
        """List all registered agents."""
        return dict(cls._registry)
    
    @classmethod
    def list_forward(cls) -> Dict[AgentType, AgentMetadata]:
        """List forward simulation agents."""
        return {
            k: v for k, v in cls._registry.items()
            if k in {AgentType.DT_POLICY, AgentType.LR_CALC_POLICY, AgentType.LR_HEUR_POLICY}
        }
    
    @classmethod
    def list_counterfactual(cls) -> Dict[AgentType, AgentMetadata]:
        """List counterfactual strategy agents."""
        return {
            k: v for k, v in cls._registry.items()
            if k.value.startswith("cf_strategy")
        }


class EnvironmentRegistry:
    """Registry for environment types."""
    
    _registry: Dict[EnvironmentType, EnvironmentMetadata] = {}
    
    @classmethod
    def register(cls, metadata: EnvironmentMetadata) -> None:
        """Register an environment type."""
        cls._registry[metadata.env_type] = metadata
    
    @classmethod
    def get(cls, env_type: EnvironmentType) -> Optional[EnvironmentMetadata]:
        """Get metadata for environment type."""
        return cls._registry.get(env_type)
    
    @classmethod
    def list_all(cls) -> Dict[EnvironmentType, EnvironmentMetadata]:
        """List all registered environments."""
        return dict(cls._registry)


# ============= Default Registration =============

def _register_defaults() -> None:
    """Register default agents and environments."""
    
    # Forward Agents
    AgentRegistry.register(AgentMetadata(
        agent_type=AgentType.DT_POLICY,
        name="HeadlessDTPolicy",
        description="Trained DT forward policy with read/retrieve modes",
        default_model_path="model_dt/simple_chi_model.zip",
    ))
    
    AgentRegistry.register(AgentMetadata(
        agent_type=AgentType.LR_CALC_POLICY,
        name="HeadlessLRCalcPolicy",
        description="Trained LR calculation policy",
        default_model_path="model_calculation/simple_chi_model.zip",
    ))
    
    AgentRegistry.register(AgentMetadata(
        agent_type=AgentType.LR_HEUR_POLICY,
        name="HeadlessLRHeurPolicy",
        description="Trained LR heuristic policy",
        default_model_path="model_heuristic/simple_chi_model.zip",
    ))
    
    # Counterfactual Strategies (from cognitive_models layer)
    AgentRegistry.register(AgentMetadata(
        agent_type=AgentType.CF_STRATEGY_DT,
        name="ZeroOutLRHeuristic (cognitive_models)",
        description="Suggest zeroing high-magnitude features to flip LR decision - from cognitive_models.counterfactual",
    ))
    
    AgentRegistry.register(AgentMetadata(
        agent_type=AgentType.CF_STRATEGY_LR_HEUR,
        name="ZeroOutLRDisplayed (cognitive_models)",
        description="Use pre-computed LR for zero-out suggestions - from cognitive_models.counterfactual",
    ))
    
    AgentRegistry.register(AgentMetadata(
        agent_type=AgentType.CF_STRATEGY_LR_DISPLAYED,
        name="ChangeDTPath (cognitive_models)",
        description="Suggest feature changes to follow different DT path - from cognitive_models.counterfactual",
    ))
    
    AgentRegistry.register(AgentMetadata(
        agent_type=AgentType.CF_STRATEGY_RECALL_DT,
        name="RecallChanges (cognitive_models)",
        description="Retrieve stored counterfactual changes from memory - from cognitive_models.counterfactual",
    ))
    
    AgentRegistry.register(AgentMetadata(
        agent_type=AgentType.CF_STRATEGY_RECALL_LR,
        name="MemoryBasedCF (cognitive_models)",
        description="Use memory to synthesize alternative decisions - from cognitive_models.counterfactual",
    ))
    
    # Environments
    EnvironmentRegistry.register(EnvironmentMetadata(
        env_type=EnvironmentType.META_ROUTER,
        name="MetaRouterEnvironment",
        description="Overall strategy selection environment (identical for forward & CF)",
    ))


# Register defaults on module import
_register_defaults()


# ============= Configuration Presets =============

COGNITIVE_PARAMS_FORWARD = {
    "retrieval_threshold": [-2.0, 0.5],
    "latency_factor": [0.0, 0.5],
    "T_enc": [0.5, 3.0],
    "T_op": [1.0, 3.0],
    "ddm_a": [0.6, 1.7],
    "ddm_s": [0.7, 1.1],
    "ddm_Tnd": 0.30,
    "ddm_norm": "l2",
    "compute_sf": 2.0,
    "lapse": 0.05,
    "chi": [0.0, 0.02],
}

COGNITIVE_PARAMS_COUNTERFACTUAL = {
    "retrieval_threshold": [-2.0, 0.5],
    "latency_factor": 1.0,
    "lapse": [0.1, 0.5],
    "over_margin": [0.0, 0.5],
    "chi": [0.0, 0.05],
}
