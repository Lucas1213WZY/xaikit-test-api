from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import generate_strategy_sample_from_params as gen


USER_STUDY_PATH = ROOT / "data/user study results/3-datasets-jan-09-2026-trials.csv"


@lru_cache(maxsize=None)
def ai_loader(dataset, xai_type):
    return gen.make_ai_loader(gen.DATASETS[dataset], xai_type)


@lru_cache(maxsize=1)
def df_params():
    return pd.read_csv(ROOT / gen.PARAMS_PATH, keep_default_na=False)


@lru_cache(maxsize=1)
def df_user():
    return gen.load_human_responses(USER_STUDY_PATH)


@lru_cache(maxsize=None)
def labels(dataset):
    return gen.class_labels(gen.DATASETS[dataset])


def candidate_instance_ids(dataset, tested):
    key = "testWithXAI" if tested == "w/ XAI" else "testWithoutXAI"
    for block_index, block in enumerate(gen.DATASETS[dataset]["blocks"]):
        for instance_id in block[key]:
            yield block_index, instance_id


def evaluate(
    *,
    dataset,
    xai_type,
    tested,
    strategy,
    instance_id,
    k=3,
    retrieval_threshold=-1.8,
    sensitivity=10.0,
    scaling_factor=3.0,
    closest_k=7,
    session=None,
    block_index=None,
):
    ds_cfg = gen.DATASETS[dataset]
    try:
        block_index, instance_number = gen.target_instance_location(ds_cfg, tested, int(instance_id), block_index)
    except ValueError:
        block_index = 0 if block_index is None else int(block_index)
        instance_number = 0
    if session is None:
        session = block_index + 1
    args = SimpleNamespace(
        dataset=dataset,
        xai_type=xai_type,
        reasoning_strategy=strategy,
        tested=tested,
        instance_number=instance_number,
        instance_id=int(instance_id),
        k=int(k),
        retrieval_threshold=float(retrieval_threshold),
        sensitivity=float(sensitivity),
        scaling_factor=float(scaling_factor),
        decay_param=0.5,
        n_sessions=1,
        closest_k=int(closest_k),
        session=int(session),
        block_index=block_index,
        seed=1234,
        train_with_explanation=True,
        user_study_path=str(USER_STUDY_PATH.relative_to(ROOT)),
        output_prefix="demo-search",
    )

    row = gen.run_strategy(
        args,
        ds_cfg,
        ai_loader(dataset, xai_type),
        gen.canonical_xai_type(xai_type),
        gen.canonical_condition(tested),
        strategy,
        int(instance_id),
    )
    nearest = gen.closest_participants(df_params(), args, ds_cfg, xai_type, tested, strategy)
    human_summary, _ = gen.response_distribution(
        df_user(), nearest, args, ds_cfg, xai_type, tested, strategy, int(instance_id)
    )
    row = {**row, **human_summary}
    row["CoAX Choice"] = gen.label_choice(row["CoAX Choice"], labels(dataset))
    row["Human Response"] = gen.label_choice(row["Human Response"], labels(dataset))
    row["AI Prediction"] = gen.label_choice(row["AI Prediction"], labels(dataset))
    row.update(
        {
            "dataset": dataset,
            "xai_type": xai_type,
            "tested": tested,
            "strategy": strategy,
            "instance_id": int(instance_id),
            "block_index": block_index,
            "instance_number": instance_number,
            "k": int(k),
            "retrieval_threshold": float(retrieval_threshold),
            "sensitivity": float(sensitivity),
            "scaling_factor": float(scaling_factor),
        }
    )
    return row


def is_match(row, min_human_n=1):
    return (
        row.get("Human n", 0) >= min_human_n
        and row.get("CoAX Choice") is not None
        and row.get("Human Response") is not None
        and row.get("CoAX Choice") == row.get("Human Response")
    )


def confidence_gap(row):
    return abs(float(row.get("CoAX Confidence") or 0) - float(row.get("Human Confidence") or 0))


def best(rows, score):
    rows = list(rows)
    return None if not rows else sorted(rows, key=score)[0]


def search_case_1():
    rows = []
    for _, instance_id in candidate_instance_ids("adult", "w/o XAI"):
        base = evaluate(dataset="adult", xai_type="None", tested="w/o XAI", strategy="Sensitive-features categorization", instance_id=instance_id, k=3, retrieval_threshold=-1.8, sensitivity=10)
        high = evaluate(dataset="adult", xai_type="None", tested="w/o XAI", strategy="Sensitive-features categorization", instance_id=instance_id, k=3, retrieval_threshold=-1.5, sensitivity=10)
        if is_match(base):
            rows.append((base, high))
    return best(rows, lambda pair: (0 if pair[0]["CoAX Choice"] != pair[1]["CoAX Choice"] else 1, -((pair[0]["CoAX Confidence"] or 0) - (pair[1]["CoAX Confidence"] or 0)), confidence_gap(pair[0])))


def search_case_2():
    rows = []
    for _, instance_id in candidate_instance_ids("wine_quality", "w/o XAI"):
        base = evaluate(dataset="wine_quality", xai_type="None", tested="w/o XAI", strategy="Sensitive-features categorization", instance_id=instance_id, k=3, retrieval_threshold=-1.8, sensitivity=4)
        high = evaluate(dataset="wine_quality", xai_type="None", tested="w/o XAI", strategy="Sensitive-features categorization", instance_id=instance_id, k=3, retrieval_threshold=-1.8, sensitivity=14)
        if is_match(base) and (high["CoAX Confidence"] or 0) > (base["CoAX Confidence"] or 0):
            rows.append((base, high))
    return best(rows, lambda pair: (confidence_gap(pair[0]), -((pair[1]["CoAX Confidence"] or 0) - (pair[0]["CoAX Confidence"] or 0))))


def search_case_3():
    rows = []
    for _, instance_id in candidate_instance_ids("adult", "w/ XAI"):
        for k in (2, 3, 4, 5):
            for sensitivity in (4, 8, 12, 16, 20):
                for retrieval_threshold in (-2.1, -1.8, -1.5):
                    salient = evaluate(dataset="adult", xai_type="Importance", tested="w/ XAI", strategy="Salient-features categorization", instance_id=instance_id, k=k, retrieval_threshold=retrieval_threshold, sensitivity=sensitivity)
                    importance = evaluate(dataset="adult", xai_type="Importance", tested="w/ XAI", strategy="Importance categorization", instance_id=instance_id, k=k, retrieval_threshold=retrieval_threshold, sensitivity=sensitivity)
                    if is_match(salient) and is_match(importance) and salient["CoAX Choice"] != importance["CoAX Choice"]:
                        rows.append((salient, importance))
    return best(rows, lambda pair: (confidence_gap(pair[0]) + confidence_gap(pair[1]), -pair[0]["Human n"] - pair[1]["Human n"]))


def search_simple_match(dataset, xai_type, tested, strategy, *, sensitivity=10, k=3, scaling_factor=3):
    rows = []
    for _, instance_id in candidate_instance_ids(dataset, tested):
        for retrieval_threshold in (-2.1, -1.8, -1.5):
            row = evaluate(dataset=dataset, xai_type=xai_type, tested=tested, strategy=strategy, instance_id=instance_id, k=k, retrieval_threshold=retrieval_threshold, sensitivity=sensitivity, scaling_factor=scaling_factor)
            if is_match(row):
                rows.append(row)
    return best(rows, lambda row: (confidence_gap(row), -row["Human n"]))


def search_case_5():
    rows = []
    for _, instance_id in candidate_instance_ids("adult", "w/ XAI"):
        for retrieval_threshold in (-2.1, -1.8, -1.5):
            for scaling_factor in (2, 3, 5, 7):
                for base_k in (2, 3, 4):
                    base = evaluate(dataset="adult", xai_type="Attribution", tested="w/ XAI", strategy="Attribution Sum", instance_id=instance_id, k=base_k, retrieval_threshold=retrieval_threshold, scaling_factor=scaling_factor)
                    if not is_match(base):
                        continue
                    flips = []
                    for alt_k in (1, 2, 3, 4, 5):
                        if alt_k == base_k:
                            continue
                        alt = evaluate(dataset="adult", xai_type="Attribution", tested="w/ XAI", strategy="Attribution Sum", instance_id=instance_id, k=alt_k, retrieval_threshold=retrieval_threshold, scaling_factor=scaling_factor)
                        if alt["CoAX Choice"] != base["CoAX Choice"]:
                            flips.append(alt)
                    if flips:
                        rows.append((base, sorted(flips, key=lambda row: abs(row["k"] - base_k))[0]))
    return best(rows, lambda pair: (confidence_gap(pair[0]), abs(pair[1]["k"] - pair[0]["k"]), -pair[0]["Human n"]))


def search_case_7():
    rows = []
    for _, instance_id in candidate_instance_ids("adult", "w/ XAI"):
        for k in (1, 2, 3, 4, 5):
            row = evaluate(dataset="adult", xai_type="Attribution", tested="w/ XAI", strategy="Attribution Sum", instance_id=instance_id, k=k, retrieval_threshold=-1.8, scaling_factor=3)
            if row.get("Human n", 0) >= 1 and row.get("CoAX Choice") != row.get("Human Response"):
                rows.append(row)
    return best(rows, lambda row: (-row["Human n"], -abs((row["CoAX Confidence"] or 0) - (row["Human Confidence"] or 0))))


def serializable(row):
    if isinstance(row, tuple):
        return [serializable(item) for item in row]
    return {key: gen.json_safe(value) for key, value in row.items()} if row else None


def main():
    found = {
        "case_1_adult_attribution_w_xai": serializable(evaluate(dataset="adult", xai_type="Attribution", tested="w/ XAI", strategy="Attribution Sum", instance_id=246, k=3, retrieval_threshold=-1.8, scaling_factor=3)),
        "case_2_adult_importance_salient": serializable(evaluate(dataset="adult", xai_type="Importance", tested="w/ XAI", strategy="Salient-features categorization", instance_id=95, k=3, retrieval_threshold=-2.3, sensitivity=4)),
        "case_2_adult_importance_switch": serializable(evaluate(dataset="adult", xai_type="Importance", tested="w/ XAI", strategy="Importance categorization", instance_id=95, k=3, retrieval_threshold=-2.3, sensitivity=4)),
        "case_3_forest_attribution_w_xai": serializable(evaluate(dataset="forest_cover", xai_type="Attribution", tested="w/ XAI", strategy="Attribution Sum", instance_id=44, k=1, retrieval_threshold=-2.3, scaling_factor=1, session=2, block_index=1)),
        "case_4_wine_attribution_without_xai": serializable(evaluate(dataset="wine_quality", xai_type="Attribution", tested="w/o XAI", strategy="Attribution Sum", instance_id=69, k=2, retrieval_threshold=-1.8, scaling_factor=5)),
        "case_4_high_threshold": serializable(evaluate(dataset="wine_quality", xai_type="Attribution", tested="w/o XAI", strategy="Attribution Sum", instance_id=69, k=2, retrieval_threshold=-1.6, scaling_factor=5)),
        "case_5_adult_none_without_xai_human_not_ai": serializable(evaluate(dataset="adult", xai_type="None", tested="w/o XAI", strategy="Sensitive-features categorization", instance_id=29, k=1, retrieval_threshold=-2.3, sensitivity=1, session=2, block_index=1)),
    }
    print(json.dumps(found, indent=2))


if __name__ == "__main__":
    main()
