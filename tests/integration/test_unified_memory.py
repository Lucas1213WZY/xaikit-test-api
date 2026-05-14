"""
Integration tests for unified memory module.

Validates both CoAX (exemplar) and CoXAM (ACT-R) backends through the unified interface.
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.cognitive_models.memory import (
    UnifiedMemory,
    MemoryConfig,
    MemoryBackend,
    Exemplar,
    Chunk,
)


def test_exemplar_backend():
    """Test CoAX exemplar-based memory backend."""
    print("\n" + "="*60)
    print("TEST 1: CoAX EXEMPLAR BACKEND")
    print("="*60)
    
    # Create memory
    memory = UnifiedMemory.create_for_coax(decay_param=0.5)
    print(f"✓ Created CoAX memory: {memory.config.backend.value}")
    
    # Store exemplars
    ex1 = Exemplar(
        label=0,
        features=np.array([1.0, 2.0, 3.0]),
        label_probs={0: 0.9, 1: 0.1},
        explanation_vector=np.array([0.5, 0.3, 0.2])
    )
    ex2 = Exemplar(
        label=1,
        features=np.array([1.1, 1.9, 3.1]),
        label_probs={0: 0.2, 1: 0.8},
        explanation_vector=np.array([0.4, 0.4, 0.2])
    )
    ex3 = Exemplar(
        label=1,
        features=np.array([5.0, 6.0, 7.0]),
        label_probs={0: 0.1, 1: 0.9},
        explanation_vector=np.array([0.1, 0.5, 0.4])
    )
    
    memory.store("ex1", ex1)
    memory.store("ex2", ex2)
    memory.store("ex3", ex3)
    print(f"✓ Stored 3 exemplars, memory size: {memory.get_size()}")
    
    # Retrieve similar exemplars
    query = np.array([1.05, 2.05, 3.05])
    results = memory.retrieve(query, k=2)
    print(f"✓ Retrieved {len(results)} exemplars:")
    for key, activation, exemplar in results:
        print(f"  - {key}: activation={activation:.4f}, label={exemplar.label}")
    
    assert len(results) == 2, "Should retrieve 2 exemplars"
    assert results[0][0] in ["ex1", "ex2"], "Should retrieve similar exemplar first"
    
    print("✓ Exemplar backend test PASSED")
    return True


def test_actr_backend():
    """Test CoXAM ACT-R-based memory backend."""
    print("\n" + "="*60)
    print("TEST 2: CoXAM ACT-R BACKEND")
    print("="*60)
    
    # Create memory
    memory = UnifiedMemory.create_for_coxam(
        decay_param=0.5,
        retrieval_threshold=-2.0,
        wm_capacity=3
    )
    print(f"✓ Created CoXAM memory: {memory.config.backend.value}")
    
    # Store chunks
    chunk1 = Chunk(
        chunk_id="c1",
        chunk_type="feature-weight",
        slots={"feature": "age", "weight": 0.6},
        creation_time=0.0
    )
    chunk2 = Chunk(
        chunk_id="c2",
        chunk_type="feature-weight",
        slots={"feature": "income", "weight": 0.8},
        creation_time=0.0
    )
    chunk3 = Chunk(
        chunk_id="c3",
        chunk_type="decision-rule",
        slots={"condition": "age > 30", "action": "approve"},
        creation_time=0.0
    )
    
    memory.store("c1", chunk1)
    memory.store("c2", chunk2)
    memory.store("c3", chunk3)
    print(f"✓ Stored 3 chunks, memory size: {memory.get_size()}")
    
    # Retrieve matching chunks
    query = {"feature": "age"}
    results = memory.retrieve(query, k=2)
    print(f"✓ Retrieved {len(results)} chunks:")
    for key, activation, chunk in results:
        print(f"  - {key}: activation={activation:.4f}, type={chunk.chunk_type}")
    
    # Test working memory
    wm = memory.get_working_memory()
    print(f"✓ Working memory contains {len(wm)} items: {wm}")
    
    # Test associative strength
    memory.add_association("c1", "c2", 1.5)
    print(f"✓ Added associative link: c1 -> c2 (strength=1.5)")
    
    print("✓ ACT-R backend test PASSED")
    return True


def test_backend_switching():
    """Test switching between backends."""
    print("\n" + "="*60)
    print("TEST 3: BACKEND SWITCHING")
    print("="*60)
    
    # Start with exemplar
    memory = UnifiedMemory.create_for_coax()
    assert memory.is_exemplar_backend(), "Should detect exemplar backend"
    print("✓ Created CoAX (exemplar) backend")
    
    # Create new memory with ACT-R
    memory = UnifiedMemory.create_for_coxam()
    assert memory.is_actr_backend(), "Should detect ACT-R backend"
    print("✓ Created CoXAM (ACT-R) backend")
    
    # Verify backend-specific methods
    actr_mem = memory.get_actr_memory()
    assert actr_mem is not None, "Should return ACT-R backend"
    print("✓ Retrieved ACT-R backend instance")
    
    exemplar_mem = memory.get_exemplar_memory()
    assert exemplar_mem is None, "Should return None for exemplar on ACT-R"
    print("✓ Verified exemplar backend is None on ACT-R memory")
    
    print("✓ Backend switching test PASSED")
    return True


def test_customizable_parameters():
    """Test parameter customization."""
    print("\n" + "="*60)
    print("TEST 4: CUSTOMIZABLE PARAMETERS")
    print("="*60)
    
    # CoAX with custom decay
    config = MemoryConfig.coax_defaults()
    config.decay_param = 0.1
    config.feature_similarity = "cosine"
    memory = UnifiedMemory(config)
    print(f"✓ Created CoAX with custom decay_param=0.1, similarity=cosine")
    
    # CoXAM with custom thresholds
    config = MemoryConfig.coxam_defaults()
    config.retrieval_threshold = -1.0
    config.wm_capacity = 6
    config.latency_factor = 100.0
    memory = UnifiedMemory(config)
    print(f"✓ Created CoXAM with:")
    print(f"  - retrieval_threshold=-1.0")
    print(f"  - wm_capacity=6")
    print(f"  - latency_factor=100.0")
    
    # Verify export
    state = memory.export_state()
    assert state["backend"] == "actr", "Should show ACT-R backend"
    assert state["context"]["wm_capacity"] == 6, "Should preserve capacity"
    print(f"✓ Exported state verification passed")
    
    print("✓ Customizable parameters test PASSED")
    return True


def test_state_management():
    """Test state export/import functionality."""
    print("\n" + "="*60)
    print("TEST 5: STATE MANAGEMENT")
    print("="*60)
    
    # Create and populate memory
    memory = UnifiedMemory.create_for_coax()
    ex1 = Exemplar(
        label=0,
        features=np.array([1.0, 2.0]),
        label_probs={0: 0.8, 1: 0.2},
        explanation_vector=np.array([0.5, 0.5])
    )
    memory.store("ex1", ex1)
    print(f"✓ Stored exemplar in memory")
    
    # Export state
    state = memory.export_state()
    print(f"✓ Exported state:")
    print(f"  - Backend: {state['backend']}")
    print(f"  - Exemplars: {state['exemplars_count']}")
    print(f"  - Decay param: {state['context']['decay_param']}")
    
    assert state["exemplars_count"] == 1, "Should have 1 exemplar"
    assert state["backend"] == "exemplar", "Should show exemplar backend"
    
    print("✓ State management test PASSED")
    return True


def test_unified_interface():
    """Test that unified interface works across backends."""
    print("\n" + "="*60)
    print("TEST 6: UNIFIED INTERFACE")
    print("="*60)
    
    # Test same operations on both backends
    
    # CoAX
    print("Testing unified interface on CoAX:")
    memory_coax = UnifiedMemory.create_for_coax()
    ex = Exemplar(label=0, features=np.array([1, 2, 3]), 
                   label_probs={0: 1.0}, explanation_vector=np.array([0, 0, 0]))
    memory_coax.store("test", ex)
    assert memory_coax.get_size() == 1
    item = memory_coax.get("test")
    assert item is not None
    print("  ✓ store(), get(), get_size() work")
    
    # CoXAM
    print("Testing unified interface on CoXAM:")
    memory_coxam = UnifiedMemory.create_for_coxam()
    chunk = Chunk(chunk_id="test", chunk_type="rule", 
                   slots={"x": 1}, creation_time=0)
    memory_coxam.store("test", chunk)
    assert memory_coxam.get_size() == 1
    item = memory_coxam.get("test")
    assert item is not None
    print("  ✓ store(), get(), get_size() work")
    
    # Clear
    memory_coax.clear()
    memory_coxam.clear()
    assert memory_coax.get_size() == 0
    assert memory_coxam.get_size() == 0
    print("  ✓ clear() works on both backends")
    
    print("✓ Unified interface test PASSED")
    return True


def run_all_tests():
    """Run all integration tests."""
    print("\n" + "█"*60)
    print("UNIFIED MEMORY - INTEGRATION TEST SUITE")
    print("█"*60)
    
    tests = [
        test_exemplar_backend,
        test_actr_backend,
        test_backend_switching,
        test_customizable_parameters,
        test_state_management,
        test_unified_interface,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
        except Exception as e:
            failed += 1
            print(f"✗ {test_func.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "█"*60)
    print(f"RESULTS: {passed} passed, {failed} failed out of {len(tests)} tests")
    print("█"*60 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
