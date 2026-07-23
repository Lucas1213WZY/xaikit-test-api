import importlib.util
from collections import Counter
from pathlib import Path


def _load_counterbalance_module():
    module_path = Path(__file__).resolve().parents[1] / "src" / "experiment_planner" / "counterbalance.py"
    spec = importlib.util.spec_from_file_location("counterbalance", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cb = _load_counterbalance_module()


def test_bradley_latin_square_balances_positions():
    conditions = ["shap", "lime", "none", "lrp"]
    orders = cb.balanced_latin_square(conditions)

    assert len(orders) == len(conditions)
    for position in range(len(conditions)):
        counts = Counter(row[position] for row in orders)
        assert counts == {condition: 1 for condition in conditions}


def test_bradley_latin_square_balances_immediate_ordered_pairs():
    conditions = ["shap", "lime", "none", "lrp"]
    orders = cb.balanced_latin_square(conditions)
    counts = Counter(
        (row[position], row[position + 1])
        for row in orders
        for position in range(len(conditions) - 1)
    )

    for left in conditions:
        for right in conditions:
            if left != right:
                assert counts[(left, right)] == 1


def test_bradley_latin_square_odd_condition_count_returns_reversed_rows():
    conditions = ["shap", "lime", "none"]
    orders = cb.balanced_latin_square(conditions)

    assert len(orders) == 2 * len(conditions)
    diagnostics = cb.counterbalancing_diagnostics(orders)
    assert diagnostics["position_balanced"] is True
    assert diagnostics["immediate_pair_balanced"] is True
    assert diagnostics["expected_immediate_pair_count"] == 2


def test_choose_counterbalancing_explicit_bradley_does_not_use_complete_for_small_n():
    conditions = ["shap", "lime", "none"]

    orders, strategy = cb.choose_counterbalancing(
        conditions,
        strategy="balanced_latin_square",
    )

    assert strategy == "balanced_latin_square_bradley_1958"
    assert len(orders) == 2 * len(conditions)


def test_choose_counterbalancing_auto_keeps_complete_for_small_n():
    conditions = ["shap", "lime", "none"]

    orders, strategy = cb.choose_counterbalancing(conditions)

    assert strategy == "complete_counterbalancing"
    assert len(orders) == 6


def test_participant_assignment_cycles_evenly_through_orders():
    orders = [["a", "b"], ["b", "a"]]
    assignments = cb.assign_participants(6, orders)

    counts = Counter(tuple(assignment["within_order"]) for assignment in assignments)
    assert counts == {("a", "b"): 3, ("b", "a"): 3}


def test_psychopy_trial_list_adapter_keeps_order_and_dict_fields():
    order = [{"xai_method": "shap"}, {"xai_method": "lime"}]

    trial_list = cb.to_psychopy_trial_list(order)

    assert trial_list == order
    assert trial_list is not order
    assert trial_list[0] is not order[0]
