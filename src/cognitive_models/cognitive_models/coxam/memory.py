import math
import random
from collections import deque

import numpy as np

MEMORY_RETRIEVAL_TIME = 0.5
MEMORY_DECAY = 0.5


class Chunk:
    def __init__(self, name, slots):
        self.name = name
        self.slots = slots
        self.retrieval_times = []
        self.prob_refreshes = []

    def add_prob_refresh(self, time, p):
        if p <= 0.0:
            return
        self.prob_refreshes.append((float(time), max(0.0, min(1.0, float(p)))))

    def update_retrieval(self, time):
        self.retrieval_times.append(time)

    def base_level_activation(self, current_time, memory_decay=MEMORY_DECAY):
        if not self.retrieval_times and not self.prob_refreshes:
            return float("-inf")

        certain = [t for t in self.retrieval_times if t < current_time]
        prob = [(t, p) for (t, p) in self.prob_refreshes if t < current_time]
        if not certain and not prob:
            return float("-inf")

        eps = 1e-10
        total = sum((current_time - t + eps) ** -memory_decay for t in certain)
        total += sum(p * (current_time - t + eps) ** -memory_decay for (t, p) in prob)
        return math.log(total) if total > 0.0 else float("-inf")

    def similarity_to(self, request, memory_mismatch_penalty=-1.0, must_match=("type", "kind")):
        for key in must_match:
            if key in request and self.slots.get(key) != request[key]:
                return float("-inf")
        return sum(memory_mismatch_penalty for key, val in request.items()
                   if key not in self.slots or self.slots[key] != val)

    def activation(self, request, current_time, memory_decay=MEMORY_DECAY, memory_mismatch_penalty=-1.0,
                   assoc_strength=0.0, cue_association_strength=2.0, fans=None):
        base = self.base_level_activation(current_time, memory_decay)
        sim = self.similarity_to(request, memory_mismatch_penalty)

        if fans and request:
            weight = 1.0 / len(request)
            for key, val in request.items():
                if key in self.slots and self.slots[key] == val:
                    fan = fans.get(val, 1)
                    if fan > 0:
                        assoc_strength += weight * (cue_association_strength - math.log(fan + 1e-10))

        return base + sim + max(0.0, assoc_strength)

    def __repr__(self):
        return f"{self.name}: {self.slots}\n"


class DeclarativeMemory:
    def __init__(self, memory_mismatch_penalty=-1.0, memory_recall_threshold=0.0,
                 cue_association_strength=2.0, memory_recall_noise=0.0):
        self.chunks = []
        self.time = 0.0
        self.memory_decay = MEMORY_DECAY
        self.memory_mismatch_penalty = memory_mismatch_penalty
        self.memory_recall_threshold = memory_recall_threshold
        self.cue_association_strength = cue_association_strength
        self.memory_recall_noise = memory_recall_noise

    def add_chunk(self, name, slots, update_retrieval=True):
        chunk = Chunk(name, slots)
        self.chunks.append(chunk)
        if update_retrieval:
            chunk.update_retrieval(self.time)
        return chunk

    def tick(self, dt=1):
        self.time += float(dt)

    def retrieve(self, request):
        best_chunk = None
        best_activation = float("-inf")

        request_values = set(request.values())
        fans = {
            val: sum(1 for chunk in self.chunks
                     if any(slot_val == val for slot_val in chunk.slots.values()))
            for val in request_values
        }

        for chunk in self.chunks:
            act = chunk.activation(
                request, self.time, self.memory_decay, self.memory_mismatch_penalty,
                assoc_strength=1.0, cue_association_strength=self.cue_association_strength, fans=fans
            )
            if self.memory_recall_noise > 0:
                eta = min(max(random.random(), 1e-10), 1.0 - 1e-10)
                act += self.memory_recall_noise * math.log((1 - eta) / eta)
            if act >= self.memory_recall_threshold and act > best_activation:
                best_activation = act
                best_chunk = chunk

        if best_chunk:
            best_chunk.update_retrieval(self.time)

        return best_chunk, best_activation, MEMORY_RETRIEVAL_TIME

    def refresh(self, chunk_name):
        chunk = self.get_chunk(chunk_name)
        if chunk:
            chunk.update_retrieval(self.time)
        return chunk

    def get_chunk(self, chunk_name):
        for chunk in self.chunks:
            if chunk.name == chunk_name:
                return chunk
        return None

    def _all_activations_for_request(self, request):
        request_values = set(request.values())
        fans = {
            val: sum(1 for chunk in self.chunks
                     if any(slot_val == val for slot_val in chunk.slots.values()))
            for val in request_values
        }
        out = [
            (
                chunk,
                chunk.activation(
                    request, self.time, self.memory_decay, self.memory_mismatch_penalty,
                    assoc_strength=1.0, cue_association_strength=self.cue_association_strength, fans=fans
                ),
            )
            for chunk in self.chunks
        ]
        out.sort(key=lambda item: item[1], reverse=True)
        return out

    def retrieval_success_prob(self, request):
        acts = self._all_activations_for_request(request)
        top_act = acts[0][1] if acts else float("-inf")
        theta = float(self.memory_recall_threshold)
        s = float(self.memory_recall_noise)
        if s <= 0.0:
            return 1.0 if top_act >= theta else 0.0
        return 1.0 / (1.0 + math.exp(-((top_act - theta) / s)))

    def __repr__(self):
        return "\n".join(
            f"{chunk} | Activation: {chunk.activation({}, self.time, self.memory_decay, self.memory_mismatch_penalty):.2f}"
            for chunk in self.chunks
        )


def _matches_request_by_slots(chunk, request):
    slots = getattr(chunk, "slots", {}) or {}
    return all(slots.get(key) == val for key, val in request.items())


class WorkingMemoryQueue:
    def __init__(self, working_memory_capacity=4):
        self.capacity = working_memory_capacity
        self._dq = deque()

    def get_by_slots(self, request):
        for chunk in reversed(self._dq):
            if _matches_request_by_slots(chunk, request):
                return chunk
        return None

    def put_chunk(self, chunk):
        name = getattr(chunk, "name", None)
        self._dq = deque([chunk_ for chunk_ in self._dq if getattr(chunk_, "name", None) != name])
        self._dq.append(chunk)
        while len(self._dq) > self.capacity:
            self._dq.popleft()

    def clear(self):
        self._dq.clear()


class CombinedMemory:
    def __init__(self, declarative_memory, working_memory_capacity=4):
        self.dm = declarative_memory
        self.wm = WorkingMemoryQueue(working_memory_capacity=working_memory_capacity)

    def add_chunk(self, name, slots, *, update_retrieval=True):
        return self.dm.add_chunk(name, slots, update_retrieval=update_retrieval)

    def tick(self, dt=1):
        self.dm.tick(dt)

    def retrieve(self, request, allow_dm=True):
        chunk = self.wm.get_by_slots(request)
        if chunk is not None:
            return chunk, None, MEMORY_RETRIEVAL_TIME
        if not allow_dm:
            return None, None, 0.0

        chunk, activation, rt = self.dm.retrieve(request)
        self.dm.tick(rt)
        if chunk is not None and _matches_request_by_slots(chunk, request):
            self.wm.put_chunk(chunk)
        return chunk, activation, rt

    def refresh(self, chunk_name):
        return self.dm.refresh(chunk_name)

    def get_chunk(self, chunk_name):
        return self.dm.get_chunk(chunk_name)

    @property
    def chunks(self):
        return self.dm.chunks

    def retrieval_success_prob(self, request):
        if self.wm.get_by_slots(request) is not None:
            return 1.0
        return float(self.dm.retrieval_success_prob(request))

    def topk_retrievals_with_prob_refresh(
        self,
        request,
        retrieval_candidate_count=3,
        memory_refresh_probability=1.0,
        add_refresh=True,
    ):
        wm_chunk = self.wm.get_by_slots(request)
        if wm_chunk is not None:
            if add_refresh and memory_refresh_probability > 0.0:
                wm_chunk.add_prob_refresh(self.dm.time, memory_refresh_probability)
            return _retrieval_result([(wm_chunk, 1.0)], 0.0)

        acts = self.dm._all_activations_for_request(request)
        theta = float(self.dm.memory_recall_threshold)
        noise = float(self.dm.memory_recall_noise)

        if not acts:
            return _retrieval_result([], 1.0)

        if noise <= 0.0:
            best_chunk, best_activation = acts[0]
            if best_activation < theta:
                return _retrieval_result([], 1.0)
            if add_refresh and memory_refresh_probability > 0.0:
                best_chunk.add_prob_refresh(self.dm.time, memory_refresh_probability)
            return _retrieval_result([(best_chunk, 1.0)], 0.0)

        kept = [(chunk, act) for chunk, act in acts if act >= theta - 100 * noise]
        if not kept:
            return _retrieval_result([], 1.0)

        logits = [(chunk, act, (act - theta) / noise) for chunk, act in kept]
        max_z = max(z for _, _, z in logits)
        none_weight = math.exp(-max_z)
        weighted = [(chunk, math.exp(z - max_z)) for chunk, _, z in logits]
        denom = none_weight + sum(weight for _, weight in weighted)
        p_none = none_weight / denom
        probs = [(chunk, weight / denom) for chunk, weight in weighted]

        if add_refresh and memory_refresh_probability > 0.0:
            for chunk, prob in probs:
                chunk.add_prob_refresh(self.dm.time, memory_refresh_probability * prob)

        probs.sort(key=lambda item: item[1], reverse=True)
        top_k = probs[:max(0, int(retrieval_candidate_count))]
        total = p_none + sum(prob for _, prob in top_k)
        if total > 0.0:
            p_none /= total
            top_k = [(chunk, prob / total) for chunk, prob in top_k]
        return _retrieval_result(top_k, p_none)

    def __repr__(self):
        return repr(self.dm)


def _retrieval_result(top_k, p_none):
    return {
        "top_k": top_k,
        "p_none": float(p_none),
        "expected_rt": MEMORY_RETRIEVAL_TIME,
        "retrieval_time": MEMORY_RETRIEVAL_TIME,
        "rt": MEMORY_RETRIEVAL_TIME,
    }


def breakdown_number_to_sf(value, max_sf):
    if value == 0 or not math.isfinite(value):
        return 1, 0, [0] * max_sf
    sign = -1 if value < 0 else 1
    ax = abs(value)
    p = math.floor(math.log10(ax))
    mant = ax / (10 ** p)
    s = f"{mant:.{max_sf - 1}f}".replace(".", "")
    if len(s) > max_sf:
        s = s[:max_sf]
        p += 1
    digits = [int(c) for c in s[:max_sf]]
    return sign, p, digits


def digits_to_value(sign, p, digits, requested_significant_figures):
    if requested_significant_figures <= 0:
        return 0.0
    s = "".join(str(d) for d in digits[:requested_significant_figures])
    mant = int(s) / (10 ** (requested_significant_figures - 1))
    return sign * mant * (10 ** p)


def remember_number_to_sf(memory, key, value, max_sf):
    sign, scale10, digits = breakdown_number_to_sf(value, max_sf)
    created = []

    memory.add_chunk(f"num:{key}:meta", {"kind": "nummeta", "key": key, "sign": sign, "p": scale10})
    created.append(f"num:{key}:meta")

    for pos, digit in enumerate(digits, start=1):
        name = f"num:{key}:d{pos}"
        memory.add_chunk(name, {"kind": "digit", "key": key, "pos": pos, "digit": digit})
        created.append(name)

    return created


def retrieve_number_to_sf(memory, key, requested_significant_figures):
    profile = build_number_profile(memory, key, requested_significant_figures)
    meta = next((value for value, prob in profile["meta"] if value is not None and prob > 0.0), None)
    if meta is None:
        return 0.0, 0, [], [], profile["expected_rt"]
    sign, p10 = meta
    digits = []
    for options in profile["digits"]:
        digit = next((value for value, prob in options if value is not None and prob > 0.0), None)
        if digit is None:
            break
        digits.append(int(digit))
    value = digits_to_value(sign, p10, digits, len(digits)) if digits else 0.0
    return value, len(digits), [], [], profile["expected_rt"]


def build_number_profile(
    memory,
    key,
    requested_significant_figures,
    *,
    retrieval_candidate_count=3,
    memory_refresh_probability=1.0,
):
    meta_dist = memory.topk_retrievals_with_prob_refresh(
        {"kind": "nummeta", "key": key},
        retrieval_candidate_count=retrieval_candidate_count,
        memory_refresh_probability=0.0,
        add_refresh=False,
    )

    meta_options = []
    meta_with_chunks = []
    p_none_meta = float(meta_dist["p_none"])
    p_meta = 1.0 - p_none_meta

    if p_none_meta > 0:
        meta_options.append((None, p_none_meta))
        meta_with_chunks.append({"value": None, "prob": p_none_meta, "chunk_name": None})

    for chunk, prob in meta_dist["top_k"]:
        sign = int(chunk.slots.get("sign", 1))
        p10 = int(chunk.slots.get("p", 0))
        chunk_name = getattr(chunk, "name", None)
        meta_options.append(((sign, p10), float(prob)))
        meta_with_chunks.append({"value": (sign, p10), "prob": float(prob), "chunk_name": chunk_name})
        if memory_refresh_probability > 0:
            chunk.add_prob_refresh(memory.dm.time, memory_refresh_probability * float(prob))

    digit_options_all = []
    digits_with_chunks = []
    prefix_prob = p_meta
    expected_rt_chain = MEMORY_RETRIEVAL_TIME

    for pos in range(1, requested_significant_figures + 1):
        digit_dist = memory.topk_retrievals_with_prob_refresh(
            {"kind": "digit", "key": key, "pos": pos},
            retrieval_candidate_count=retrieval_candidate_count,
            memory_refresh_probability=0.0,
            add_refresh=False,
        )
        opts_legacy = []
        opts_with_chunks = []
        p_none = float(digit_dist["p_none"])
        p_hit = 1.0 - p_none

        if p_none > 0:
            opts_legacy.append((None, p_none))
            opts_with_chunks.append({"value": None, "prob": p_none, "chunk_name": None})

        for chunk, prob in digit_dist["top_k"]:
            digit = int(chunk.slots.get("digit", 0))
            chunk_name = getattr(chunk, "name", None)
            opts_legacy.append((digit, float(prob)))
            opts_with_chunks.append({"value": digit, "prob": float(prob), "chunk_name": chunk_name})
            if memory_refresh_probability > 0 and prefix_prob > 0:
                chunk.add_prob_refresh(memory.dm.time, memory_refresh_probability * prefix_prob * float(prob))

        digit_options_all.append(opts_legacy)
        digits_with_chunks.append(opts_with_chunks)
        expected_rt_chain += MEMORY_RETRIEVAL_TIME
        prefix_prob *= p_hit

    return {
        "meta": meta_options,
        "digits": digit_options_all,
        "expected_rt": expected_rt_chain,
        "meta_with_chunks": meta_with_chunks,
        "digits_with_chunks": digits_with_chunks,
    }
