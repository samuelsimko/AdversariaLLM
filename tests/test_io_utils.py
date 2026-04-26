"""Tests for shared I/O helpers: dict matching, nested lookups, JSON encoder, error decorator."""
import json
import logging

import pytest

from adversariallm.errors import print_exceptions
from adversariallm.io_utils.data_analysis import (
    get_nested_value,
    normalize_value_for_grouping,
)
from adversariallm.io_utils.database import check_match
from adversariallm.io_utils.json_utils import CompactJSONEncoder


# ---------------------------------------------------------------------------
# check_match
# ---------------------------------------------------------------------------


def test_check_match_primitive_equality():
    assert check_match("gcg", "gcg") is True
    assert check_match("gcg", "pair") is False
    assert check_match(42, 42) is True
    assert check_match(None, None) is True


def test_check_match_list_filter_is_any_of():
    # A list on the filter side means "any of these".
    assert check_match("gcg", ["gcg", "pair"]) is True
    assert check_match("autodan", ["gcg", "pair"]) is False


def test_check_match_set_filter_is_any_of():
    assert check_match("gcg", {"gcg", "pair"}) is True


def test_check_match_list_on_both_sides_requires_equality():
    # Per the current implementation, iterable filter against iterable doc requires exact equality.
    assert check_match([1, 2, 3], [1, 2, 3]) is True
    assert check_match([1, 2, 3], [3, 2, 1]) is False
    assert check_match([1, 2], [1, 2, 3]) is False


def test_check_match_nested_dict_recurses():
    doc = {
        "model": "llama",
        "attack_params": {"num_steps": 100, "seed": 0},
        "dataset_params": {"idx": 5},
    }
    assert check_match(doc, {"model": "llama"}) is True
    assert check_match(doc, {"attack_params": {"num_steps": 100}}) is True
    assert check_match(doc, {"attack_params": {"num_steps": 999}}) is False


def test_check_match_missing_filter_key_fails():
    doc = {"model": "llama"}
    assert check_match(doc, {"missing_key": "x"}) is False


def test_check_match_dict_filter_against_non_dict_doc_fails():
    assert check_match("llama", {"model": "llama"}) is False


def test_check_match_combines_list_any_with_nested_dict():
    doc = {"attack": "gcg", "attack_params": {"num_steps": 100}}
    assert check_match(doc, {"attack": ["gcg", "pair"], "attack_params": {"num_steps": 100}}) is True


# ---------------------------------------------------------------------------
# get_nested_value
# ---------------------------------------------------------------------------


def test_get_nested_value_returns_leaf():
    data = {"a": {"b": {"c": 42}}}
    assert get_nested_value(data, ["a", "b", "c"]) == 42


def test_get_nested_value_returns_default_for_missing_key():
    data = {"a": {"b": 1}}
    assert get_nested_value(data, ["a", "missing"]) == "unknown"
    assert get_nested_value(data, ["a", "missing"], default=None) is None


def test_get_nested_value_returns_default_when_intermediate_is_not_dict():
    data = {"a": 42}
    assert get_nested_value(data, ["a", "b"]) == "unknown"


def test_get_nested_value_with_empty_path_returns_root():
    data = {"a": 1}
    assert get_nested_value(data, []) == data


# ---------------------------------------------------------------------------
# normalize_value_for_grouping
# ---------------------------------------------------------------------------


def test_normalize_value_for_grouping_coerces_integer_floats():
    assert normalize_value_for_grouping(1.0) == 1
    assert normalize_value_for_grouping(0.0) == 0
    # Non-integer floats are preserved
    assert normalize_value_for_grouping(0.5) == 0.5


def test_normalize_value_for_grouping_preserves_ints_and_strings():
    assert normalize_value_for_grouping(7) == 7
    assert normalize_value_for_grouping("gcg") == "gcg"


def test_normalize_value_for_grouping_recurses_through_containers():
    value = {"nums": [1.0, 2.0, 3.5], "meta": {"seed": 0.0}}
    normalized = normalize_value_for_grouping(value)
    assert normalized == {"nums": [1, 2, 3.5], "meta": {"seed": 0}}


def test_normalize_value_for_grouping_preserves_tuple_type():
    normalized = normalize_value_for_grouping((1.0, 2.0))
    assert isinstance(normalized, tuple)
    assert normalized == (1, 2)


# ---------------------------------------------------------------------------
# CompactJSONEncoder
# ---------------------------------------------------------------------------


def test_compact_encoder_puts_small_dict_on_single_line():
    encoded = json.dumps({"a": 1, "b": 2}, cls=CompactJSONEncoder)
    assert "\n" not in encoded
    assert json.loads(encoded) == {"a": 1, "b": 2}


def test_compact_encoder_puts_int_list_on_single_line_regardless_of_length():
    big = list(range(2000))
    encoded = json.dumps(big, cls=CompactJSONEncoder)
    # No newlines inside the big int list (single-line shortcut)
    assert encoded.count("\n") == 0
    assert json.loads(encoded) == big


def test_compact_encoder_splits_long_string_lists():
    long_strings = ["x" * 100 for _ in range(5)]  # pushes str repr past 200 chars
    encoded = json.dumps(long_strings, cls=CompactJSONEncoder)
    assert "\n" in encoded
    assert json.loads(encoded) == long_strings


def test_compact_encoder_round_trips_nested_structure():
    payload = {
        "config": {"model": "llama", "seed": 0},
        "steps": [{"step": i, "scores": {"p": [0.1, 0.2]}} for i in range(3)],
    }
    encoded = json.dumps(payload, cls=CompactJSONEncoder)
    assert json.loads(encoded) == payload


# ---------------------------------------------------------------------------
# print_exceptions decorator
# ---------------------------------------------------------------------------


def test_print_exceptions_rerases_original_exception(capsys):
    @print_exceptions
    def boom():
        raise RuntimeError("kaboom")

    with pytest.raises(RuntimeError, match="kaboom"):
        boom()

    captured = capsys.readouterr()
    # traceback.print_exception writes to stderr
    assert "RuntimeError" in captured.err
    assert "kaboom" in captured.err


def test_print_exceptions_passthrough_success():
    @print_exceptions
    def add(a, b):
        return a + b

    assert add(2, 3) == 5


def test_print_exceptions_preserves_wrapped_name():
    @print_exceptions
    def my_function():
        return 1

    assert my_function.__name__ == "my_function"


def test_print_exceptions_exposes_wrapped_attributes_for_hydra():
    # Hydra inspects f.__code__ to detect the calling file; the wrapper must expose it.
    @print_exceptions
    def my_function():
        return 1

    assert hasattr(my_function, "__code__")
