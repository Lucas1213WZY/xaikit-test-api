#!/usr/bin/env python3
"""
Standalone test runner for cr_agent module tests.
Does not require pytest - uses simple assertions.
"""

import sys
import numpy as np
from pathlib import Path

# Add project to path
test_dir = Path(__file__).parent
cr_agent_dir = test_dir.parent
src_dir = cr_agent_dir.parent
project_root = src_dir.parent

sys.path.insert(0, str(project_root))
sys.path.insert(0, str(src_dir))


def test_module_imports():
    """Test that all modules can be imported."""
    print("\n" + "=" * 70)
    print("TEST 1: Module Imports")
    print("=" * 70)
    
    try:
        from src.cognitive_models.cr_agent.headless_policies import (
            HeadlessDTPolicy,
            HeadlessLRCalcPolicy,
            HeadlessLRHeurPolicy,
        )
        from src.cognitive_models.cr_agent.forward_meta_router import (
            load_forward_strategies,
            run_meta_on_batch,
        )
        from src.cognitive_models.cr_agent.counterfactual_meta_router import (
            CounterfactualMetaRouter,
            load_counterfactual_strategies,
        )
        from src.cognitive_models.cr_agent.interface import CRAgentRunner, MetaRunner
        from src.cognitive_models.cr_agent.registry import (
            COGNITIVE_PARAMS_FORWARD,
            COGNITIVE_PARAMS_COUNTERFACTUAL,
        )
        print("✅ All modules imported successfully")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_strategy_factory():
    """Test strategy factory creation."""
    print("\n" + "=" * 70)
    print("TEST 2: Strategy Loading (load_counterfactual_strategies)")
    print("=" * 70)
    
    try:
        from src.cognitive_models.cr_agent.counterfactual_meta_router import load_counterfactual_strategies
        
        strategies = load_counterfactual_strategies()
        
        # Verify it's a dict
        assert isinstance(strategies, dict), "load_counterfactual_strategies() should return dict"
        assert len(strategies) > 0, "Strategies dict should not be empty"
        
        print(f"✅ Loaded {len(strategies)} counterfactual strategies:")
        for name in sorted(strategies.keys()):
            print(f"  - {name}")
        
        # Verify each strategy has cognitive_models API methods
        for name, strategy in strategies.items():
            assert hasattr(strategy, "infer"), \
                f"Strategy {name} missing infer method"
            assert hasattr(strategy, "feedback"), \
                f"Strategy {name} missing feedback method"
            assert hasattr(strategy, "new_instance"), \
                f"Strategy {name} missing new_instance method"
        
        print("✅ All strategies have infer(), feedback(), new_instance() methods")
        print("✅ All strategies loaded from cognitive_models API")
        return True
    
    except Exception as e:
        print(f"⚠️  Strategy loading test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_strategy_loading():
    """Test forward strategy loading."""
    print("\n" + "=" * 70)
    print("TEST 3: Forward Strategy Loading (load_forward_strategies)")
    print("=" * 70)
    
    try:
        from src.cognitive_models.cr_agent.forward_meta_router import load_forward_strategies
        
        # Use factory to get already-configured strategies
        strategies = load_forward_strategies()
        
        assert isinstance(strategies, dict), "load_forward_strategies() should return dict"
        assert len(strategies) > 0, "Strategies dict should not be empty"
        
        print(f"✅ Loaded {len(strategies)} forward strategies:")
        for name in sorted(strategies.keys()):
            print(f"  - {name}")
        
        # Verify each strategy has cognitive_models API methods
        for name, strategy in strategies.items():
            assert hasattr(strategy, "infer"), \
                f"Strategy {name} missing infer method"
            assert hasattr(strategy, "feedback"), \
                f"Strategy {name} missing feedback method"
            assert hasattr(strategy, "new_instance"), \
                f"Strategy {name} missing new_instance method"
            print(f"  ✓ {name}: infer(), feedback(), new_instance() verified")
        
        print("✅ All forward strategies load from cognitive_models API")
        
        return True
    
    except Exception as e:
        print(f"⚠️  Forward strategy loading test: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_registry():
    """Test registry presets."""
    print("\n" + "=" * 70)
    print("TEST 4: Registry Presets")
    print("=" * 70)
    
    try:
        from src.cognitive_models.cr_agent.registry import (
            COGNITIVE_PARAMS_FORWARD,
            COGNITIVE_PARAMS_COUNTERFACTUAL,
        )
        
        # Check forward
        assert isinstance(COGNITIVE_PARAMS_FORWARD, dict), \
            "COGNITIVE_PARAMS_FORWARD should be dict"
        assert len(COGNITIVE_PARAMS_FORWARD) > 0, \
            "COGNITIVE_PARAMS_FORWARD should not be empty"
        
        print(f"✅ COGNITIVE_PARAMS_FORWARD has {len(COGNITIVE_PARAMS_FORWARD)} keys")
        for key in sorted(COGNITIVE_PARAMS_FORWARD.keys())[:3]:
            print(f"  - {key}")
        if len(COGNITIVE_PARAMS_FORWARD) > 3:
            print(f"  ... and {len(COGNITIVE_PARAMS_FORWARD) - 3} more")
        
        # Check counterfactual
        assert isinstance(COGNITIVE_PARAMS_COUNTERFACTUAL, dict), \
            "COGNITIVE_PARAMS_COUNTERFACTUAL should be dict"
        assert len(COGNITIVE_PARAMS_COUNTERFACTUAL) > 0, \
            "COGNITIVE_PARAMS_COUNTERFACTUAL should not be empty"
        
        print(f"✅ COGNITIVE_PARAMS_COUNTERFACTUAL has {len(COGNITIVE_PARAMS_COUNTERFACTUAL)} keys")
        
        return True
    
    except Exception as e:
        print(f"⚠️  Registry test: {e}")
        return False


def test_interface_classes():
    """Test interface classes."""
    print("\n" + "=" * 70)
    print("TEST 5: Interface Classes (CRAgentRunner, MetaRunner)")
    print("=" * 70)
    
    try:
        from src.cognitive_models.cr_agent.interface import CRAgentRunner, MetaRunner
        
        # Check CRAgentRunner
        assert hasattr(CRAgentRunner, "run_forward_episode"), \
            "CRAgentRunner missing run_forward_episode"
        assert hasattr(CRAgentRunner, "run_counterfactual_episode"), \
            "CRAgentRunner missing run_counterfactual_episode"
        
        print("✅ CRAgentRunner has required methods:")
        print("  - run_forward_episode")
        print("  - run_counterfactual_episode")
        
        # Check MetaRunner
        assert hasattr(MetaRunner, "run_episode"), \
            "MetaRunner missing run_episode"
        assert hasattr(MetaRunner, "__init__"), \
            "MetaRunner missing __init__"
        
        print("✅ MetaRunner has required methods:")
        print("  - __init__")
        print("  - run_episode")
        
        return True
    
    except Exception as e:
        print(f"⚠️  Interface test: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("CR_AGENT STANDALONE TEST SUITE")
    print("=" * 70)
    
    results = {
        "Module Imports": test_module_imports(),
        "Strategy Loading": test_strategy_factory(),
        "Forward Strategy Loading": test_forward_strategy_loading(),
        "Registry": test_registry(),
        "Interface Classes": test_interface_classes(),
    }
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, passed_flag in results.items():
        status = "✅ PASS" if passed_flag else "⚠️  SKIP"
        print(f"{status}: {test_name}")
    
    print("\n" + "=" * 70)
    print(f"Total: {passed}/{total} tests passed")
    print("=" * 70 + "\n")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
