"""
ACT-R-based memory backend for CoXAM.

Implements probabilistic activation using Base-Level Learning (BLL),
associative strength, and partial matching for chunk retrieval.
"""

from typing import List, Optional, Tuple, Any, Dict, Union, Set
import math
import numpy as np
from collections import defaultdict, deque
from datetime import datetime
from .interface import MemoryInterface, Chunk, ReasoningContext, MemoryBackend
from .utils import (
    base_level_learning, compute_similarity_activation, add_activation_noise,
    compute_retrieval_latency, compute_chunk_similarity, get_timestamp_diff
)


class ACTRMemory(MemoryInterface):
    """
    ACT-R-based probabilistic memory backend for CoXAM.
    
    Storage: Hierarchical chunks with BLL tracking
    Retrieval: Probabilistic based on activation
    Activation: BLL + Associative Strength + Partial Matching
    
    Key characteristics:
    - Chunks store structured knowledge (slots)
    - Base-Level Learning (BLL): ln(sum(t_i^-d))
    - Associative strength: w_j * s_ji (spreading activation)
    - Partial matching: mismatch penalties
    - Stochastic retrieval with latency variability
    """
    
    def __init__(self, context: ReasoningContext):
        super().__init__(context)
        self.chunks: Dict[str, Chunk] = {}
        self.chunk_retrievals: Dict[str, List[float]] = defaultdict(list)
        self.activation = dict()
        self.time = context.current_time or 0.0

        # Expose latency_exponent as direct attribute (default 1.0 if not in context)
        self.latency_exponent: float = getattr(context, 'latency_exponent', 1.0)

        # Associative links: source_chunk -> {target_chunk -> strength}
        self.associations: Dict[str, Dict[str, float]] = defaultdict(dict)

        # Working memory queue for active chunks (key deque)
        self.working_memory: deque = deque(maxlen=context.wm_capacity)

        # Validate backend
        if self.context.backend != MemoryBackend.ACTR:
            raise ValueError(f"ACTRMemory requires backend=ACTR, got {self.backend}")

    # ---- convenience properties matching old DeclarativeMemory attribute names ----

    @property
    def latency_factor(self) -> float:
        return self.context.latency_factor

    @property
    def retrieval_threshold(self) -> float:
        return self.context.retrieval_threshold

    @property
    def activation_noise(self) -> float:
        return self.context.activation_noise

    @property
    def decay(self) -> float:
        return self.context.decay_param
    
    def store(self, key: str, value: Union[Chunk, None]) -> None:
        """
        Store a chunk in declarative memory.
        
        Args:
            key: Unique chunk identifier
            value: Chunk object
        """
        if not isinstance(value, Chunk):
            raise TypeError(f"ACTRMemory.store() expects Chunk, got {type(value)}")
        
        self.chunks[key] = value
        self.chunk_retrievals[key] = [self.time]  # Initial encoding
        
        # Add to working memory
        self.working_memory.append(key)
    
    # ================================================================
    # Correct BLL and slot-activation helpers
    # ================================================================

    def _chunk_bll(self, key: str) -> float:
        """
        BLL = ln( sum_{t_j < now}(now - t_j)^-d  +  sum_m p_m*(now - t_m*)^-d )
        Uses actual retrieval timestamps from chunk_retrievals plus mean-field
        contribution from chunk.prob_refreshes.
        """
        current_time = self.time
        decay = self.context.decay_param
        chunk = self.chunks.get(key)
        if chunk is None:
            return float("-inf")

        retrieval_times = self.chunk_retrievals.get(key, [])
        certain = [t for t in retrieval_times if t < current_time]
        prob = [(t_s, p) for (t_s, p) in chunk.prob_refreshes if t_s < current_time]

        if not certain and not prob:
            return float("-inf")

        eps = 1e-10
        s_certain = sum((current_time - t + eps) ** -decay for t in certain)
        s_prob = sum(p * (current_time - t_s + eps) ** -decay for (t_s, p) in prob)
        total = s_certain + s_prob
        if total <= 0.0:
            return float("-inf")
        return math.log(total)

    def _chunk_similarity(
        self,
        chunk: Chunk,
        request: dict,
        must_match: tuple = ('type', 'kind'),
    ) -> float:
        """
        ACT-R slot similarity: hard must_match rejection + mismatch penalty per unmatched slot.
        Returns -inf if any must_match slot fails; otherwise sums -mismatch_penalty for each
        request slot whose value is absent or different in the chunk.
        """
        for k in must_match:
            if k in request and chunk.slots.get(k) != request[k]:
                return float("-inf")
        sim = 0.0
        for key, val in request.items():
            if key not in chunk.slots or chunk.slots[key] != val:
                sim -= self.context.mismatch_penalty
        return sim

    def _all_activations_for_request(self, request: dict) -> List[Tuple[Chunk, float]]:
        """
        Compute (chunk, activation) for every chunk, sorted descending.
        Activation = BLL + slot_similarity + fan-weighted associativity.
        Chunks that fail must_match are excluded.
        """
        request_values = set(v for v in request.values() if v is not None)
        fans: Dict[Any, int] = {}
        for val in request_values:
            fans[val] = sum(
                1 for ch in self.chunks.values()
                if any(sv == val for sv in ch.slots.values())
            )

        out: List[Tuple[Chunk, float]] = []
        for key, chunk in self.chunks.items():
            sim = self._chunk_similarity(chunk, request)
            if sim == float("-inf"):
                continue

            base = self._chunk_bll(key)

            assoc = 0.0
            if fans and request:
                w = 1.0 / len(request)
                for req_k, val in request.items():
                    if req_k in chunk.slots and chunk.slots[req_k] == val:
                        fan = fans.get(val, 1)
                        if fan > 0:
                            s_ji = self.context.max_assoc_strength - math.log(fan + 1e-10)
                            assoc += w * s_ji
            assoc = max(0.0, assoc)

            out.append((chunk, base + sim + assoc))

        out.sort(key=lambda t: t[1], reverse=True)
        return out

    def retrieve(self, query: Any, k: int = 1,
                 similarity_threshold: Optional[float] = None) -> List[Tuple[str, float, Chunk]]:
        """
        Retrieve top-k chunks matching query using correct BLL activation + slot similarity.

        Returns:
            List of (key, activation_score, chunk) tuples, sorted by activation desc
        """
        if not self.chunks:
            return []

        if similarity_threshold is None:
            similarity_threshold = self.context.retrieval_threshold

        if isinstance(query, Chunk):
            request = query.slots
        elif isinstance(query, dict):
            request = query
        else:
            raise TypeError(f"Query must be Chunk or dict, got {type(query)}")

        all_acts = self._all_activations_for_request(request)

        results = []
        for chunk, activation in all_acts:
            activation = add_activation_noise(activation, self.context.activation_noise)
            if activation < similarity_threshold:
                continue
            results.append((chunk.chunk_id, activation, chunk))

        results.sort(key=lambda x: x[1], reverse=True)

        for key, _, _ in results[:k]:
            self.chunk_retrievals[key].append(self.time)

        return results[:k]
    
    def retrieve_with_latency(self, query: Any, k: int = 1) -> Tuple[List[Tuple[str, Chunk]], float]:
        """
        Retrieve chunks and compute access latency.
        
        ACT-R latency: RT = F * exp(-activation)
        
        Returns:
            Tuple of ([(key, chunk), ...], latency_ms)
        """
        retrieved = self.retrieve(query, k)
        
        if not retrieved:
            return [], 0.0
        
        # Use highest activation for latency
        best_activation = retrieved[0][1]
        latency = compute_retrieval_latency(best_activation, self.context.latency_factor)
        
        result = [(key, chunk) for key, _, chunk in retrieved]
        return result, latency
    
    def get(self, key: str) -> Optional[Chunk]:
        """Get a specific chunk by key."""
        return self.chunks.get(key)
    
    def update_activation(self, key: str, increase: float) -> None:
        """
        Update chunk activation on retrieval.
        
        In ACT-R, this updates BLL by recording a new retrieval time.
        """
        if key in self.chunks:
            # Record retrieval at current time
            self.chunk_retrievals[key].append(self.time)
            
            # Update chunk's activation tracking
            chunk = self.chunks[key]
            chunk.reference_count += 1
            new_activation = base_level_learning(chunk.reference_count, self.context.decay_param)
            chunk.activations.append(new_activation)
    
    def clear(self) -> None:
        """Clear all chunks and associations."""
        self.chunks.clear()
        self.chunk_retrievals.clear()
        self.associations.clear()
        self.working_memory.clear()
        self.activation.clear()
    
    def get_size(self) -> int:
        """Get number of stored chunks."""
        return len(self.chunks)
    
    def export_state(self) -> Dict[str, Any]:
        """Export memory state for inspection/debugging."""
        return {
            "backend": self.backend.value,
            "current_time": self.time,
            "chunks_count": len(self.chunks),
            "chunks": {
                key: {
                    "chunk_type": chunk.chunk_type,
                    "creation_time": chunk.creation_time,
                    "reference_count": chunk.reference_count,
                    "activations_count": len(chunk.activations),
                    "slots_keys": list(chunk.slots.keys())
                }
                for key, chunk in self.chunks.items()
            },
            "associations_count": sum(len(v) for v in self.associations.values()),
            "working_memory_size": len(self.working_memory),
            "context": {
                "decay_param": self.context.decay_param,
                "retrieval_threshold": self.context.retrieval_threshold,
                "latency_factor": self.context.latency_factor,
                "wm_capacity": self.context.wm_capacity,
                "mismatch_penalty": self.context.mismatch_penalty
            }
        }
    
    def import_state(self, state: Dict[str, Any]) -> None:
        """Import memory state (restoration)."""
        self.time = state.get("current_time", 0.0)
        # Note: Restoring chunks requires access to original slot data
    
    def add_association(self, source_key: str, target_key: str, strength: float) -> None:
        """
        Add associative link between chunks.
        
        Args:
            source_key: Chunk that activates the association
            target_key: Target chunk that benefits from association
            strength: Associative strength (typically 0-2)
        """
        strength = min(strength, self.context.max_assoc_strength)
        self.associations[source_key][target_key] = strength
    
    def update_time(self, new_time: float) -> None:
        """Update internal time (used for BLL decay calculations)."""
        self.time = new_time
    
    def _compute_partial_match(self, query_slots: Dict[str, Any], 
                                chunk_slots: Dict[str, Any]) -> float:
        """
        Compute partial matching bonus/penalty.
        
        Compares query slots to chunk slots, applying mismatch penalties
        for non-matches and similarity bonuses for matches.
        
        Returns:
            Matching score (can be negative if many mismatches)
        """
        if not query_slots:
            return 0.0
        
        slot_weights = {}  # Equal weights by default
        similarity = compute_chunk_similarity(query_slots, chunk_slots, slot_weights)
        
        # Convert similarity [0,1] to activation impact
        # Full match (1.0) -> bonus, partial/no match -> penalty
        matching_bonus = similarity * self.context.max_assoc_strength
        
        # Penalty for query slots not in chunk
        query_only = set(query_slots.keys()) - set(chunk_slots.keys())
        penalty = len(query_only) * self.context.mismatch_penalty
        
        return matching_bonus - penalty
    
    def get_working_memory(self) -> List[str]:
        """Get IDs of chunks currently in working memory."""
        return list(self.working_memory)
    
    def get_chunk_history(self, key: str) -> Dict[str, Any]:
        """Get detailed history for a chunk."""
        if key not in self.chunks:
            return {}

        chunk = self.chunks[key]
        retrievals = self.chunk_retrievals.get(key, [])

        return {
            "chunk_id": key,
            "chunk_type": chunk.chunk_type,
            "creation_time": chunk.creation_time,
            "retrieval_times": retrievals,
            "retrieval_count": len(retrievals),
            "activation_history": chunk.activations
        }

    # ================================================================
    # Old-API compatibility helpers (matches DeclarativeMemory interface)
    # ================================================================

    def add_chunk(self, name: str, slots: dict, *, update_retrieval: bool = True) -> Chunk:
        """Create and store a Chunk by name (matches old DeclarativeMemory.add_chunk)."""
        chunk = Chunk(
            chunk_id=name,
            chunk_type=slots.get("type", slots.get("kind", "generic")),
            slots=slots,
            creation_time=self.time,
        )
        self.chunks[name] = chunk
        self.chunk_retrievals[name] = [self.time] if update_retrieval else []
        self.working_memory.append(name)
        return chunk

    def tick(self, dt: float = 1) -> None:
        """Advance internal time (matches old DeclarativeMemory.tick)."""
        self.time += float(dt)

    def get_chunk(self, name: str) -> Optional[Chunk]:
        """Retrieve chunk by name/id (matches old DeclarativeMemory.get_chunk)."""
        return self.chunks.get(name)

    def refresh(self, name: str) -> Optional[Chunk]:
        """Record a deterministic retrieval at current time for named chunk."""
        chunk = self.chunks.get(name)
        if chunk is not None:
            self.chunk_retrievals[name].append(self.time)
        return chunk

    # ================================================================
    # Top-k probabilistic retrieval distribution
    # ================================================================

    def topk_retrievals_with_prob_refresh(
        self,
        request: dict,
        k: int = 3,
        refresh_prob: float = 1.0,
        add_refresh: bool = True,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Softmax retrieval distribution over all chunks (with a 'none' retrieval option).

        Softmax with none:
            z_i = (A_i - theta) / s
            p_i   = exp(z_i - max_z) / (exp(-max_z) + sum_j exp(z_j - max_z))
            p_none = exp(-max_z)     / (exp(-max_z) + sum_j exp(z_j - max_z))

        Expected RT:
            E[RT] = p_none * F*exp(-fexp*theta)  +  sum_i p_i * F*exp(-fexp*A_i)

        When add_refresh is True, logs a prob_refresh on each chunk proportional to p_i,
        recording the expected activation boost from being part of this retrieval distribution.

        Returns:
            {
                "top_k": [(chunk, prob), ...],   # top-k renormalized
                "p_none": float,
                "expected_rt": float,
            }
        """
        F = self.context.latency_factor
        fexp = self.latency_exponent
        theta = self.context.retrieval_threshold
        s = self.context.activation_noise

        safe_theta_rt = F * math.exp(-fexp * theta) if math.isfinite(theta) else 0.0

        acts = self._all_activations_for_request(request)
        if not acts:
            return {"top_k": [], "p_none": 1.0, "expected_rt": safe_theta_rt}

        # --- deterministic fallback ---
        if s <= 0.0:
            best_chunk, best_A = acts[0]
            if best_A >= theta:
                if add_refresh and refresh_prob > 0.0:
                    best_chunk.add_prob_refresh(self.time, refresh_prob)
                return {
                    "top_k": [(best_chunk, 1.0)],
                    "p_none": 0.0,
                    "expected_rt": F * math.exp(-fexp * best_A),
                }
            return {"top_k": [], "p_none": 1.0, "expected_rt": safe_theta_rt}

        # Filter extreme outliers (> 100 sigma below threshold)
        acts = [(ch, A) for (ch, A) in acts if A >= theta - 100 * s]
        if not acts:
            return {"top_k": [], "p_none": 1.0, "expected_rt": safe_theta_rt}

        # Numerically-stabilized softmax with none option
        logits = [(ch, A, (A - theta) / s) for (ch, A) in acts]
        max_z = max(z for _, _, z in logits)
        exp_z = [(ch, A, math.exp(z - max_z)) for (ch, A, z) in logits]
        sum_exp = sum(ez for _, _, ez in exp_z)

        p_none_raw = math.exp(-max_z)
        denom = p_none_raw + sum_exp

        p_none = p_none_raw / denom
        probs: List[Tuple[Chunk, float, float]] = [(ch, A, ez / denom) for (ch, A, ez) in exp_z]

        # Expected retrieval time
        expected_rt = p_none * safe_theta_rt
        for ch, A, p in probs:
            expected_rt += p * F * math.exp(-fexp * A)

        # Log probabilistic refreshes
        if add_refresh and refresh_prob > 0.0:
            for ch, _, p in probs:
                pr = refresh_prob * p
                if pr > 0.0:
                    ch.add_prob_refresh(self.time, pr)

        # Select top-k and renormalize slice
        probs_sorted = sorted(probs, key=lambda t: t[2], reverse=True)
        top_k_raw = [(ch, p) for (ch, _, p) in probs_sorted[:max(0, k)]]

        total_p = p_none + sum(p for _, p in top_k_raw)
        if total_p > 0.0:
            top_k = [(ch, p / total_p) for ch, p in top_k_raw]
            p_none = p_none / total_p
        else:
            top_k, p_none = top_k_raw, 0.0

        if verbose:
            print(f"[topk] p_none={p_none:.3f}, expected_rt={expected_rt:.4f}")
            for ch, p in top_k:
                print(f"  {ch.chunk_id}: p={p:.3f}")

        return {"top_k": top_k, "p_none": p_none, "expected_rt": expected_rt}
