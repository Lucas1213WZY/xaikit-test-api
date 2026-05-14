"""
QUICK REFERENCE - Unified Memory API

Copy-paste templates and common usage patterns
"""

# ============================================================================
# IMPORTS
# ============================================================================

from src.cognitive_models.memory import (
    UnifiedMemory, MemoryConfig, MemoryBackend,
    Exemplar, Chunk, ReasoningContext
)
import numpy as np


# ============================================================================
# TEMPLATE 1: Create CoAX Memory
# ============================================================================

# Basic
memory = UnifiedMemory.create_for_coax()

# Custom decay
memory = UnifiedMemory.create_for_coax(decay_param=0.2)

# Custom similarity metric
memory = UnifiedMemory.create_for_coax(feature_similarity="cosine")

# Multiple customizations
memory = UnifiedMemory.create_for_coax(
    decay_param=0.1,
    feature_similarity="euclidean"
)


# ============================================================================
# TEMPLATE 2: Create CoXAM Memory
# ============================================================================

# Basic
memory = UnifiedMemory.create_for_coxam()

# Custom working memory size
memory = UnifiedMemory.create_for_coxam(wm_capacity=8)

# Custom retrieval threshold
memory = UnifiedMemory.create_for_coxam(retrieval_threshold=-0.5)

# Full customization
memory = UnifiedMemory.create_for_coxam(
    decay_param=0.5,
    retrieval_threshold=-1.5,
    latency_factor=100.0,
    activation_noise=0.1,
    wm_capacity=6,
    mismatch_penalty=2.0,
    max_assoc_strength=2.0
)


# ============================================================================
# TEMPLATE 3: Store Items (CoAX)
# ============================================================================

# Create exemplar
exemplar = Exemplar(
    label=0,
    features=np.array([1.0, 2.0, 3.0]),
    label_probs={0: 0.9, 1: 0.1},
    explanation_vector=np.array([0.5, 0.3, 0.2])
)

# Store
memory.store("exemplar_key_1", exemplar)


# ============================================================================
# TEMPLATE 4: Store Items (CoXAM)
# ============================================================================

# Create chunk
chunk = Chunk(
    chunk_id="chunk_001",
    chunk_type="feature-weight",
    slots={
        "feature": "age",
        "weight": 0.6,
        "importance": "high"
    },
    creation_time=0.0
)

# Store
memory.store("chunk_key_1", chunk)


# ============================================================================
# TEMPLATE 5: Retrieve Items (CoAX)
# ============================================================================

# Retrieve top-3
query_features = np.array([1.05, 2.05, 3.05])
results = memory.retrieve(query_features, k=3)

# Process results
for key, activation, exemplar in results:
    print(f"Key: {key}")
    print(f"  Activation: {activation:.4f}")
    print(f"  Label: {exemplar.label}")
    print(f"  Features: {exemplar.features}")


# ============================================================================
# TEMPLATE 6: Retrieve Items (CoXAM)
# ============================================================================

# Retrieve top-3 with latency
query = {"feature": "age"}
retrieved, latency_ms = memory.retrieve_with_latency(query, k=3)

# Process results
print(f"Latency: {latency_ms:.2f}ms")
for key, chunk in retrieved:
    print(f"Key: {key}")
    print(f"  Type: {chunk.chunk_type}")
    print(f"  Slots: {chunk.slots}")


# ============================================================================
# TEMPLATE 7: Retrieve Single Best Item
# ============================================================================

# CoAX or CoXAM - works on both
best_item = memory.retrieve_top_item(query)
if best_item:
    print(f"Best match: {best_item}")


# ============================================================================
# TEMPLATE 8: Get Specific Item
# ============================================================================

item = memory.get("exemplar_key_1")
if item:
    print(item)


# ============================================================================
# TEMPLATE 9: Update Activation (Reinforcement)
# ============================================================================

# When exemplar/chunk is successfully used
memory.update_activation("exemplar_key_1", increase=0.5)


# ============================================================================
# TEMPLATE 10: CoXAM-Specific: Add Associative Links
# ============================================================================

# Add spreading activation link (CoXAM only)
memory.add_association(
    source_key="chunk_1",
    target_key="chunk_2",
    strength=1.5
)


# ============================================================================
# TEMPLATE 11: CoXAM-Specific: Update Time
# ============================================================================

# Progress internal time (affects BLL decay)
memory.update_time(10.0)


# ============================================================================
# TEMPLATE 12: Get Working Memory Contents
# ============================================================================

# CoXAM only
active_chunks = memory.get_working_memory()
print(f"Working memory: {active_chunks}")


# ============================================================================
# TEMPLATE 13: State Export (Inspection)
# ============================================================================

state = memory.export_state()

print(f"Backend: {state['backend']}")
print(f"Items: {state.get('exemplars_count', state.get('chunks_count'))}")
print(f"Context: {state['context']}")


# ============================================================================
# TEMPLATE 14: Dynamic Reconfiguration
# ============================================================================

# Change parameters at runtime
memory.reconfigure(decay_param=0.1, wm_capacity=8)


# ============================================================================
# TEMPLATE 15: Clear All Memory
# ============================================================================

memory.clear()
print(f"Memory size after clear: {memory.get_size()}")


# ============================================================================
# TEMPLATE 16: Check Backend Type
# ============================================================================

if memory.is_exemplar_backend():
    print("Using CoAX exemplar backend")
elif memory.is_actr_backend():
    print("Using CoXAM ACT-R backend")


# ============================================================================
# TEMPLATE 17: Get Backend Instance
# ============================================================================

# Get backend if you need backend-specific methods
if memory.is_exemplar_backend():
    exemplar_mem = memory.get_exemplar_memory()
    all_exemplars = exemplar_mem.get_exemplars()
    
elif memory.is_actr_backend():
    actr_mem = memory.get_actr_memory()
    wm = actr_mem.get_working_memory()


# ============================================================================
# TEMPLATE 18: Create with MemoryConfig (Advanced)
# ============================================================================

# Create config manually for more control
config = MemoryConfig(
    backend=MemoryBackend.ACTR,
    decay_param=0.5,
    retrieval_threshold=-1.5,
    latency_factor=100.0,
    wm_capacity=6,
    current_time=0.0
)

memory = UnifiedMemory(config)


# ============================================================================
# TEMPLATE 19: Function Parameter Patterns
# ============================================================================

def run_reasoning(query_data, memory_system="coax", k=5):
    """Example function using unified memory"""
    
    # Create memory based on parameter
    if memory_system == "coax":
        memory = UnifiedMemory.create_for_coax()
    else:
        memory = UnifiedMemory.create_for_coxam()
    
    # Retrieve
    results = memory.retrieve(query_data, k=k)
    
    # Process (works on both)
    return [(key, score) for key, score, _ in results]


# ============================================================================
# TEMPLATE 20: Batch Processing
# ============================================================================

def batch_retrieve(memory, queries, k=3):
    """Retrieve results for multiple queries"""
    all_results = []
    for query_id, query in enumerate(queries):
        results = memory.retrieve(query, k=k)
        all_results.append({
            'query_id': query_id,
            'results': results
        })
    return all_results


# ============================================================================
# PARAMETER REFERENCE
# ============================================================================

"""
CoAX (Exemplar) Parameters:
  - decay_param: 0.1-1.0 (lower = faster decay)
  - feature_similarity: "euclidean" (default) or "cosine"

CoXAM (ACT-R) Parameters:
  - decay_param: 0.1-1.0 (BLL exponent, lower = faster decay)
  - retrieval_threshold: -5.0 to 5.0 (lower = easier retrieval)
  - latency_factor: 50-200 (ms scaling, 0 = no latency)
  - activation_noise: 0.0-1.0 (0 = deterministic)
  - wm_capacity: 2-16 (working memory size)
  - mismatch_penalty: 0.5-2.0 (higher = stricter matching)
  - max_assoc_strength: 0.5-3.0 (ceiling for links)
"""


# ============================================================================
# ACTIVATION FORMULAS
# ============================================================================

"""
CoAX Activation:
  activation = temporal_decay(time_since_encoding) × similarity(query, exemplar)
  
  Where: temporal_decay(t) = 1 / (1 + decay_param × t)
         similarity uses euclidean or cosine distance

CoXAM Activation:
  activation = BLL + ∑(associative_strength) + partial_match - penalties
  
  Where: BLL = ln(∑(t_i ^ -decay_param))  [Base-Level Learning]
         associative_strength from working memory sources
         partial_match based on slot-to-slot similarity
         penalties for slots not in query

  Retrieval Latency: RT = latency_factor × exp(-activation)
"""
