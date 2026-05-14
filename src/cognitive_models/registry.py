"""
Strategy Registry - Auto-discovery and plugin management.

Provides dynamic strategy registration, discovery, and instantiation.
Strategies are automatically discovered from module imports and can be
retrieved by name for use in API endpoints and orchestrators.
"""

from typing import Dict, Optional, Type, Any, List
import inspect
from .interface import ReasoningStrategy, StrategyConfig, StrategyMetadata, StrategyType


class StrategyRegistry:
    """
    Central registry for all reasoning strategies.
    
    Features:
    - Auto-discovery: Scans imported modules for Strategy classes
    - Registration: Store strategy classes by name
    - Instantiation: Create strategy instances with config
    - Validation: Check strategy availability
    - Introspection: List available strategies with metadata
    
    Usage:
        StrategyRegistry.register("sensitive_features", SensitiveFeatures)
        strategy = StrategyRegistry.get("sensitive_features", config)
        all_strategies = StrategyRegistry.list_strategies()
    """
    
    _strategies: Dict[str, Type[ReasoningStrategy]] = {}
    _metadata: Dict[str, StrategyMetadata] = {}
    _initialized: bool = False
    
    @classmethod
    def register(cls, name: str, strategy_class: Type[ReasoningStrategy], 
                 metadata: Optional[StrategyMetadata] = None) -> None:
        """
        Register a strategy class by name.
        
        Args:
            name: Unique strategy identifier
            strategy_class: Strategy class (must implement ReasoningStrategy)
            metadata: Optional StrategyMetadata (extracted from class if not provided)
        """
        if not issubclass(strategy_class, ReasoningStrategy):
            raise TypeError(f"{strategy_class.__name__} must inherit from ReasoningStrategy")
        
        cls._strategies[name] = strategy_class
        
        # Extract metadata from class if not provided
        if metadata is None:
            # Try to get metadata from strategy class
            if hasattr(strategy_class, 'get_metadata'):
                metadata = strategy_class.get_metadata()
            else:
                # Fallback: Create minimal metadata
                metadata = StrategyMetadata(
                    name=name,
                    display_name=name.replace('_', ' ').title(),
                    strategy_type=StrategyType.COAX_FORWARD,
                    description=f"Strategy: {name}",
                    category="Unknown"
                )
        
        cls._metadata[name] = metadata
    
    @classmethod
    def unregister(cls, name: str) -> bool:
        """
        Unregister a strategy.
        
        Args:
            name: Strategy identifier
            
        Returns:
            bool: True if strategy was registered and removed
        """
        if name in cls._strategies:
            del cls._strategies[name]
            if name in cls._metadata:
                del cls._metadata[name]
            return True
        return False
    
    @classmethod
    def get(cls, name: str, config: StrategyConfig) -> ReasoningStrategy:
        """
        Instantiate and return a strategy by name.
        
        Args:
            name: Strategy identifier
            config: StrategyConfig with parameters
            
        Returns:
            ReasoningStrategy instance
            
        Raises:
            KeyError: If strategy not registered
            TypeError: If instantiation fails
        """
        if name not in cls._strategies:
            available = list(cls._strategies.keys())
            raise KeyError(f"Strategy '{name}' not found. Available: {available}")
        
        strategy_class = cls._strategies[name]
        
        try:
            # Pass config to constructor
            # Different strategies may accept different parameters
            return strategy_class(config)
        except TypeError as e:
            raise TypeError(f"Failed to instantiate {name}: {e}")
    
    @classmethod
    def exists(cls, name: str) -> bool:
        """
        Check if a strategy is registered.
        
        Args:
            name: Strategy identifier
            
        Returns:
            bool: True if strategy exists
        """
        return name in cls._strategies
    
    @classmethod
    def validate(cls, name: str) -> Optional[str]:
        """
        Validate strategy name and return error message if invalid.
        
        Args:
            name: Strategy identifier
            
        Returns:
            str: Error message if invalid, None if valid
        """
        if not name or not isinstance(name, str):
            return "Strategy name must be a non-empty string"
        
        if name not in cls._strategies:
            available = list(cls._strategies.keys())
            return f"Strategy '{name}' not registered. Available: {available}"
        
        return None
    
    @classmethod
    def list_strategies(cls) -> Dict[str, Dict[str, Any]]:
        """
        List all registered strategies with metadata.
        
        Returns:
            Dict mapping strategy names to info dicts:
              {
                'strategy_name': {
                  'display_name': str,
                  'description': str,
                  'type': str,                    # 'coax_forward', 'coxam_forward', etc.
                  'category': str,                # 'CoAX', 'CoXAM', etc.
                  'supported_modes': [str, ...],  # ['retrieve', 'read', etc.]
                  'parameters': {str: Any}        # Parameter hints
                }
              }
        """
        result = {}
        for name, metadata in cls._metadata.items():
            result[name] = {
                'display_name': metadata.display_name,
                'description': metadata.description,
                'type': metadata.strategy_type.value,
                'category': metadata.category,
                'supported_modes': [m.value for m in metadata.supported_modes],
                'parameters': metadata.parameters
            }
        return result
    
    @classmethod
    def list_by_type(cls, strategy_type: StrategyType) -> Dict[str, Dict[str, Any]]:
        """
        List strategies of a specific type.
        
        Args:
            strategy_type: Filter by this type (e.g., COAX_FORWARD)
            
        Returns:
            Dict mapping strategy names to metadata dicts
        """
        result = {}
        for name, metadata in cls._metadata.items():
            if metadata.strategy_type == strategy_type:
                result[name] = {
                    'display_name': metadata.display_name,
                    'description': metadata.description,
                    'category': metadata.category,
                    'parameters': metadata.parameters
                }
        return result
    
    @classmethod
    def discover_from_module(cls, module: Any) -> List[str]:
        """
        Auto-discover and register all ReasoningStrategy subclasses in a module.
        
        Args:
            module: Python module to scan
            
        Returns:
            List of registered strategy names
        """
        registered = []
        
        # Scan module members for Strategy classes
        for name, obj in inspect.getmembers(module, inspect.isclass):
            # Skip abstract classes and the base interface
            if name.startswith('_') or obj is ReasoningStrategy:
                continue
            
            # Check if it's a Strategy subclass
            if issubclass(obj, ReasoningStrategy) and obj is not ReasoningStrategy:
                strategy_name = name.lower()
                
                try:
                    cls.register(strategy_name, obj)
                    registered.append(strategy_name)
                except Exception as e:
                    print(f"Warning: Failed to register {name}: {e}")
        
        return registered
    
    @classmethod
    def initialize(cls) -> None:
        """
        Initialize registry by auto-discovering strategies from submodules.
        
        Should be called once at module import time.
        Imports forward and counterfactual strategy modules.
        """
        if cls._initialized:
            return
        
        try:
            # Discover strategies from forward modules
            from . import forward
            if hasattr(forward, 'coax_forward_rs'):
                cls.discover_from_module(forward.coax_forward_rs)
            if hasattr(forward, 'coxam_forward_rs'):
                cls.discover_from_module(forward.coxam_forward_rs)
            
            # Discover strategies from counterfactual modules
            from . import counterfactual
            if hasattr(counterfactual, 'coxam_counterfactual_rs'):
                cls.discover_from_module(counterfactual.coxam_counterfactual_rs)
        
        except ImportError as e:
            print(f"Warning: Could not auto-discover strategies: {e}")
        
        cls._initialized = True
    
    @classmethod
    def reset(cls) -> None:
        """Clear all registered strategies (for testing)."""
        cls._strategies.clear()
        cls._metadata.clear()
        cls._initialized = False
    
    @classmethod
    def get_count(cls) -> int:
        """Return number of registered strategies."""
        return len(cls._strategies)
    
    @classmethod
    def get_names(cls) -> List[str]:
        """Return list of all registered strategy names."""
        return list(cls._strategies.keys())
