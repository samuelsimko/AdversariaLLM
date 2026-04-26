"""Tests for the PromptDataset registry and index-selection helper."""
import pytest
import torch
from omegaconf import OmegaConf

from adversariallm.dataset.prompt_dataset import PromptDataset


def _make_config(**kwargs):
    return OmegaConf.create(kwargs)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_register_and_from_name_round_trip():
    @PromptDataset.register("_test_dataset_registry_roundtrip")
    class _DS(PromptDataset):
        pass

    assert PromptDataset.from_name("_test_dataset_registry_roundtrip") is _DS


def test_from_name_raises_value_error_for_unknown_name():
    with pytest.raises(ValueError, match="Unknown dataset"):
        PromptDataset.from_name("__definitely_not_registered__")


def test_register_is_a_decorator_that_returns_the_class():
    decorator = PromptDataset.register("_test_registry_returns_class")

    class _DS(PromptDataset):
        pass

    assert decorator(_DS) is _DS


# ---------------------------------------------------------------------------
# _select_idx
# ---------------------------------------------------------------------------


class _Probe(PromptDataset):
    """Exposes _select_idx for direct testing."""


def test_select_idx_with_none_returns_full_range():
    config = _make_config(idx=None)
    ds = _Probe(config)
    idx, config_idx = ds._select_idx(config, 5)
    assert idx.tolist() == [0, 1, 2, 3, 4]
    assert config_idx is None


def test_select_idx_with_int_slices_single_element():
    config = _make_config(idx=3)
    ds = _Probe(config)
    idx, config_idx = ds._select_idx(config, 10)
    assert idx.tolist() == [3]
    assert config_idx == 3


def test_select_idx_with_sequence_selects_those_indices():
    config = _make_config(idx=[1, 3, 4])
    ds = _Probe(config)
    idx, _ = ds._select_idx(config, 10)
    assert idx.tolist() == [1, 3, 4]


def test_select_idx_parses_list_range_string():
    config = _make_config(idx="list(range(2, 5))")
    ds = _Probe(config)
    idx, config_idx = ds._select_idx(config, 10)
    assert idx.tolist() == [2, 3, 4]
    assert config_idx == [2, 3, 4]


def test_select_idx_rejects_malformed_string():
    config = _make_config(idx="[0, 1, 2]")
    ds = _Probe(config)
    with pytest.raises(ValueError, match="Does not start with"):
        ds._select_idx(config, 10)


def test_select_idx_rejects_invalid_type():
    config = _make_config(idx=1.5)
    ds = _Probe(config)
    with pytest.raises(ValueError, match="Invalid idx"):
        ds._select_idx(config, 10)


def test_select_idx_shuffle_is_deterministic_per_seed():
    cfg_a = _make_config(idx=None, shuffle=True, seed=42)
    cfg_b = _make_config(idx=None, shuffle=True, seed=42)
    cfg_c = _make_config(idx=None, shuffle=True, seed=7)

    ds = _Probe(cfg_a)
    idx_a, _ = ds._select_idx(cfg_a, 10)
    idx_b, _ = ds._select_idx(cfg_b, 10)
    idx_c, _ = ds._select_idx(cfg_c, 10)

    assert torch.equal(idx_a, idx_b)
    assert not torch.equal(idx_a, idx_c)


def test_select_idx_shuffle_plus_slice_composes():
    config = _make_config(idx=[0, 1, 2], shuffle=True, seed=0)
    ds = _Probe(config)
    idx, _ = ds._select_idx(config, 10)
    # Three elements taken from the permuted index — whatever seed=0 produces
    assert idx.shape == (3,)
    assert idx.dtype == torch.int64
